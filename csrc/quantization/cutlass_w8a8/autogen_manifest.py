from dataclasses import dataclass
from typing import List, Tuple
from itertools import product
import copy

## Utilities
def min_tuples(a, b):
    return (min(a[0], b[0]),
            min(a[1], b[1]),
            min(a[2], b[2]))

@dataclass
class Cutlass2xArgs:
    arch: int
    tile_shape: Tuple[int, int, int]
    warp_shape: Tuple[int, int, int]
    instruction_shape: Tuple[int, int, int]
    thread_block_swizzle: str
    gemm_mode: str
    main_loop_stages: int

    def with_tile_shape(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_shape = ts 
        return clone

    def with_warp_shape(self, ws):
        clone = copy.deepcopy(self)
        clone.warp_shape = ws
        return clone

    def with_instruction_shape(self, inst):
        clone = copy.deepcopy(self)
        clone.instruction_shape = inst
        return clone

    def with_gemm_mode(self, gemm_mode):
        clone = copy.deepcopy(self)
        clone.gemm_mode = gemm_mode 
        return clone

    def with_main_loop_stages(self, mls):
        clone = copy.deepcopy(self)
        clone.main_loop_stages = mls 
        return clone

@dataclass
class Cutlass3xArgs:
    dtype_str: str
    arch: int
    tile_shape: Tuple[int, int, int]
    cluster_shape: Tuple[int, int, int]
    kernel_schedule: str
    epilogue_schedule: str
    tile_schedule: str
    gemm_mode: str

    def with_tile_shape(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_shape = ts 
        return clone

    def with_cluster_shape(self, cs):
        clone = copy.deepcopy(self)
        clone.tile_shape = cs 
        return clone

    def with_tile_schedule(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_schedule = ts 
        return clone

    def with_kernel_schedule(self, ks):
        clone = copy.deepcopy(self)
        clone.kernel_schedule = ks 
        return clone

    def with_epilogue_schedule(self, es):
        clone = copy.deepcopy(self)
        clone.epilogue_schedule = es 
        return clone

    def with_gemm_mode(self, gm):
        clone = copy.deepcopy(self)
        clone.gemm_mode = gm 
        return clone

    def with_dtype_str(self, dtype_str):
        clone = copy.deepcopy(self)
        clone.dtype_str = dtype_str 
        return clone

Cutlass2xArgsList = [
        Cutlass2xArgs(80,
                      (128, 128, 64),
                      (64, 64, 64),
                      (16, 8, 32),
                      "cutlass::gemm::threadblock::ThreadblockSwizzleStreamK",
                      "cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel",
                      5)]
