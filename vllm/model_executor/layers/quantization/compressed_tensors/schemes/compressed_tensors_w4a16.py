from typing import Callable, List, Tuple, Union, Optional

import torch
from torch.nn import Parameter

from vllm._C import ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinState, marlin_permute_scales, get_scale_perms
from vllm.model_executor.utils import set_weight_attrs

from enum import Enum
import numpy
import torch.nn.functional as F

__all__ = ["CompressedTensorsW4A16"]


class CompressedTensorsW4A16(CompressedTensorsScheme):

    def __init__(self, strategy: str, group_size: Optional[int] = None):
        self.strategy = strategy
        self.group_size = group_size

        if self.strategy == "group" and self.group_size is None:
            raise ValueError(
                "group_size must be given when using strategy group")

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       layer_name: str, **kwargs):

        pack_factor = 8  # the only one we support for now
        # for group size, 128 things next to each other in memory, 2nd dimension for things next to each other in memory
        output_size_per_partition = sum(output_partition_sizes)

        if self.group_size is not None:
            group_size = self.group_size
        else:
            group_size = input_size

        weight_scale_dim = None
        scales_and_zp_size = input_size // group_size

        if input_size != input_size_per_partition and self.group_size is not None:
            weight_scale_dim = 1
            scales_and_zp_size = input_size_per_partition // group_size

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "packed_dim": 1,
            "pack_factor": 8
        })
        set_weight_attrs(weight, {"weight_loader": weight_loader})
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        set_weight_attrs(weight_scale, {"weight_loader": weight_loader})
        set_weight_attrs(weight_scale, {
            "input_dim": weight_scale_dim,
            "output_dim": 0
        })
        layer.register_parameter("weight_scale", weight_scale)

        weight_zero_point = Parameter(
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        set_weight_attrs(weight_zero_point, {"weight_loader": weight_loader})
        set_weight_attrs(weight_zero_point, {
            "input_dim": weight_scale_dim,
            "output_dim": 0
        })
        layer.register_parameter("weight_zero_point", weight_zero_point)

        weight_shape = Parameter(torch.empty(2,
                                             device="cuda",
                                             dtype=torch.int64),
                                 requires_grad=False)

        layer.register_parameter("weight_shape", weight_shape)
        set_weight_attrs(weight_shape, {"weight_loader": weight_loader})

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        layer.input_size = input_size
        layer.marlin_state = GPTQMarlinState.REPACK
        layer.is_k_full = True
        layer.group_size = group_size

        max_workspace_size = (output_size_per_partition // 64) * 16

        workspace = torch.zeros(max_workspace_size,
                                dtype=torch.int,
                                requires_grad=False)
        layer.workspace = workspace

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        reshaped_x = x.reshape(-1, x.shape[-1])

        size_m = reshaped_x.shape[0]
        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        full_size_k = layer.input_size

        out_shape = x.shape[:-1] + (part_size_n, )

        if layer.marlin_state == GPTQMarlinState.REPACK:
            layer.marlin_state = GPTQMarlinState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                # It is important to use resize_() here since it ensures
                # the same buffer is reused
                getattr(layer, name).resize_(new_t.shape)
                getattr(layer, name).copy_(new_t)
                del new_t

            cur_device = layer.weight.device

            # Reset g_idx related tensors
            layer.g_idx = Parameter(torch.empty(0,
                                                dtype=torch.int,
                                                device=cur_device),
                                    requires_grad=False)
            layer.g_idx_sort_indices = Parameter(torch.empty(
                0, dtype=torch.int, device=cur_device),
                                                 requires_grad=False)

            # Repack weights
            marlin_qweight = ops.gptq_marlin_repack(
                layer.weight.t().contiguous(), layer.g_idx_sort_indices,
                part_size_k, part_size_n, 4)

            replace_tensor("weight", marlin_qweight)

            # Permute scales
            scales_size_k = part_size_k
            scales_size_n = part_size_n

            marlin_scales = marlin_permute_scales(
                layer.weight_scale.squeeze().t().contiguous(), scales_size_k,
                scales_size_n, layer.group_size, 4)
            replace_tensor("weight_scale", marlin_scales)

        output = ops.gptq_marlin_gemm(reshaped_x, layer.weight,
                                      layer.weight_scale, layer.g_idx,
                                      layer.g_idx_sort_indices,
                                      layer.workspace, 4, size_m, part_size_n,
                                      part_size_k, layer.is_k_full)

        return output.reshape(out_shape)
