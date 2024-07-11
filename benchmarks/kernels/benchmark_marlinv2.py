import argparse
import copy
import itertools
import pickle as pkl
import time
import math
from typing import Callable, Iterable, List, Tuple
from functools import partial

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

from vllm._custom_classes import VLLMType
from vllm import vllm_type
from vllm import _custom_ops as ops
from vllm.utils import FlexibleArgumentParser
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_weights_into_int32, gptq_pack, quantize_weights)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MarlinWorkspace, 
)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    marlin_permute_scales, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
)

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())[1:]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]


def marlinv2_pack_weights(w_q: torch.tensor, wtype: VLLMType) -> torch.tensor:
    w_q = pack_weights_into_int32(w_q, wtype)
    return ops.marlinv2_prepack_B(w_q, wtype)


def make_bench_tensors(
        atype: torch.dtype, wtype: VLLMType, group_size: int,
        m: int, n: int, k: int
) -> Tuple[torch.tensor, List[Tuple[torch.tensor, torch.tensor, torch.tensor]]]:
    assert wtype.is_integer(), "TODO: support floating point weights"

    # we want to make sure that weights don't fit into L2 cache between runs so 
    #  we construct enough weights to exceed L2 cache, which is 50mb on a H100
    #  so we target total weight size > 2*50mb
    num_weights = math.ceil(2 * 50 * 1024**2 * 8 / (k * n * wtype.size_bits))

    a = torch.randn((m, k), device="cuda", dtype=atype) * 5
    weights = [torch.randn((k, n), device="cuda", dtype=atype) 
               for _ in range(num_weights)]
    quanitized_weights = [quantize_weights(w, wtype, group_size) 
                          for w in weights]

    return a, quanitized_weights


# impl


# bench
def bench_fn(
    label: str,
    sub_label: str,
    description: str,
    fn: Callable,
) -> TMeasurement:

    min_run_time = 1
    return TBenchmark.Timer(
        stmt="fn()",
        globals={ "fn": fn },
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)
    
def loop_over_weights(
    a: torch.tensor, 
    weights: List[Tuple[torch.tensor, torch.tensor, torch.tensor]],
    fn: Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor], None]
):
    for w_ref, w_q, w_s in weights:
        fn(a, w_ref, w_q, w_s)


def bench(atype: torch.dtype, wtype: VLLMType, group_size: int, m: int, k: int,
          n: int, label: str, sub_label: str, benchmark_marlinv1: bool = True,
          benchmark_marlinv2_best: bool = True
    ) -> Iterable[TMeasurement]:
    a, weights = make_bench_tensors(atype, wtype, group_size, m, n, k)
    sub_label += f", L={len(weights)}"

    weights_marlinv2 = [(w_ref, marlinv2_pack_weights(w_q, wtype), w_s)
                    for w_ref, w_q, w_s in weights]


    timers = []
    # pytorch impl
    timers.append(
        bench_fn(
            label,
            sub_label,
            "torch.matmul",
            lambda: loop_over_weights(
                a, weights, 
                lambda a, w_ref, w_q, w_s: torch.matmul(a, w_ref),
            )
        ))

    if benchmark_marlinv1:
        w_ref = weights[0][0]
        
        sort_indices = torch.empty(0, dtype=torch.int, device=w_ref.device)
        g_idx = torch.empty(0, dtype=torch.int, device=w_ref.device)
        
        def marlinv1_pack_weights(w_q: torch.tensor) -> torch.tensor:
            w_q_gptq = gptq_pack(w_q, wtype.size_bits, *w_ref.shape)
            return ops.gptq_marlin_repack(
                w_q_gptq, sort_indices, *w_ref.shape, wtype.size_bits)
        
        def marlinv1_permute_scales(w_s: torch.tensor) -> torch.tensor:
            return marlin_permute_scales(
                w_s, *w_ref.shape, group_size, wtype.size_bits)
        
        weights_marlinv1 = [(w_ref, marlinv1_pack_weights(w_q), 
                             marlinv1_permute_scales(w_s))
                            for w_ref, w_q, w_s in weights]
            
        workspace = MarlinWorkspace(
            w_ref.shape[1], GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL)
        
        # marlinv1
        timers.append(
            bench_fn(
                label,
                sub_label,
                "marlin_orig",
                lambda: loop_over_weights(
                    a, weights_marlinv1, 
                    lambda a, w_ref, w_q, w_s:
                        ops.gptq_marlin_gemm(a, w_q, w_s, g_idx, sort_indices,
                        workspace.scratch, wtype.size_bits, size_m=a.shape[0], 
                        size_n=w_ref.shape[1], size_k=w_ref.shape[0], 
                        is_k_full=True)
                    )
                )
            )

    # marlinv2
    timers.append(
        bench_fn(
            label,
            sub_label,
            "marlinv2_heuristic",
            lambda: loop_over_weights(
                a, weights_marlinv2, 
                lambda a, _, w_q, w_s:
                    ops.marlinv2_gemm(a, w_q, wtype, 
                                      b_scales=w_s, b_group_size=group_size)
                )
            )
        )

    if benchmark_marlinv2_best:
        print("Finding best schedule for marlinv2")
        best = None
        best_schedule = None
        for schedule in ops.marlinv2_supported_schedules(wtype):
            res = bench_fn(
                label,
                sub_label,
                f"marlinv2_best",
                lambda: loop_over_weights(
                    a, weights_marlinv2, 
                    lambda a, _, w_q, w_s:
                        ops.marlinv2_gemm(a, w_q, wtype, 
                                          b_scales=w_s, b_group_size=group_size,
                                          schedule=schedule)
                    )
            )
            print(f"  {res.median:5.5} ", schedule)
            if not best or res.median < best.median:
                best = res
                best_schedule = schedule
        print("Best schedule:", best_schedule)
        timers.append(best)

    return timers


# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:

    results = []
    for m, k, n in MKNs:
        timers = bench(dtype, vllm_type.s4, 128, m, k, n,  f"{dtype}-gemm",
                       f"MKN=({m}x{k}x{n})")
        print_timers(timers)
        results.extend(timers)

    return results


# output makers
def make_output(
    data: Iterable[TMeasurement],
    MKNs: Iterable[Tuple[int, int, int]],
    base_description: str,
    timestamp=None,
):

    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


# argparse runners


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args.dtype, MKNs)

    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args.dtype, MKNs)

    make_output(data, MKNs, f"range_bench-{args.dtype}")


def run_model_bench(args):

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args.dtype, MKNs)
        model_bench_data.append(data)

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {args.dtype} {model}-TP{tp_size} ====")
        print_timers(data)

    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)


if __name__ == "__main__":

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float16
        raise ValueError("unsupported dtype")

    parser = FlexibleArgumentParser(
        description="""
Benchmark Cutlass GEMM.

    To run square GEMMs:
        python3 ./benchmarks/kernels/benchmark_marlinv2.py --dtype float16 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/kernels/benchmark_marlinv2.py --dtype float16 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/kernels/benchmark_marlinv2.py --dtype float16 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dtype",
        type=to_torch_dtype,
        required=True,
        help="Available options are ['int8', 'fp8']",
    )
    subparsers = parser.add_subparsers(dest="cmd")

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--dim-start", type=int, required=True)
    range_parser.add_argument("--dim-end", type=int, required=True)
    range_parser.add_argument("--dim-increment", type=int, required=True)
    range_parser.add_argument("--m-constant", type=int, default=None)
    range_parser.add_argument("--n-constant", type=int, default=None)
    range_parser.add_argument("--k-constant", type=int, default=None)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
