from typing import Any, Dict, List, Optional, Tuple, Union

import os
import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

class FP8Config(QuantizationConfig):
    """Config class for FP8."""
    def __init__(
        self,
        scheme: str,
    ) -> None:
        assert scheme == "static" or scheme == "dynamic"
        self.scheme = scheme

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
        scheme = cls.get_from_keys(config, ["scheme"])
        return cls(scheme=scheme)

    def get_linear_method(self) -> "FP8LinearMethod":
        return FP8LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

class FP8LinearMethod(LinearMethodBase):
    """Linear method for StaticFP8
    .
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: FP8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size, params_dtype
        
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.float8_e4m3fn),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)

        if len(output_partition_sizes) == 1:
            weight_scale = Parameter(
                torch.empty(len(output_partition_sizes), dtype=torch.float32),
                requires_grad=False
            )
            layer.register_parameter("weight_scale", weight_scale)
            set_weight_attrs(weight_scale, extra_weight_attrs)
            set_weight_attrs(weight_scale, {
                "shard_indexer": self.scales_shard_indexer,
            })
        else:
            weight_scale = Parameter(
                torch.empty(sum(output_partition_sizes), dtype=torch.float32),
                requires_grad=False
            )
            layer.register_parameter("weight_scale", weight_scale)
            set_weight_attrs(weight_scale, extra_weight_attrs)
            set_weight_attrs(weight_scale, {
                "shard_indexer": self.scales_shard_indexer_NKK,
                "logical_widths": output_partition_sizes
            })

        if self.quant_config.scheme == "static":
            act_scale = Parameter(
                torch.empty(len(output_partition_sizes), dtype=torch.float32), 
                requires_grad=False
            )
            layer.register_parameter("act_scale", act_scale)
            set_weight_attrs(act_scale, extra_weight_attrs)
            set_weight_attrs(act_scale, {
                "shard_indexer": self.scales_shard_indexer,
            })

        layer.logical_widths = output_partition_sizes

    def shard_id_as_int(
        self, 
        shard_id: Union[str, int]
    ) -> int:
        if isinstance(shard_id, int):
            return shard_id
        assert isinstance(shard_id, str)
        qkv_idxs = { "q": 0, "k": 1, "v": 2 }
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    def scales_shard_indexer_NKK(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: Union[str, int],
        logical_widths,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_id = self.shard_id_as_int(shard_id)
        offset = sum(logical_widths[:shard_id]) 
        size = logical_widths[shard_id]
        # update loaded weight with copies for broadcast.
        loaded_weight = loaded_weight.repeat(size)
        return param[offset : offset + size], loaded_weight
    
    def scales_shard_indexer(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: Union[str, int],
        logical_widths,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del logical_widths
        return param[self.shard_id_as_int(shard_id)], loaded_weight

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Quantize Input
        if self.quant_config.scheme == "dynamic":
            qinput, x_scale = per_tensor_quantize_dyanmic(x)

        elif self.quant_config.scheme == "static":
            # empirically, these are all the same
            x_scale = layer.act_scale.max()
            qinput = per_tensor_quantize_static(x, x_scale)

        # NON LOGICAL SHARDED MATRIX
        if len(layer.logical_widths) == 1:
            out, _ = torch._scaled_mm(
                qinput,
                layer.weight.t(),
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                bias=bias,
            )
            return out  

        # LOGICAL SHARDED MATRIC
        # FOR LOOP TO BE REPLACED BY CUTLASS KERNEL W/ EPILOGUE FUSION
        else:
            output = torch.zeros(x.shape[0], layer.weight.shape[0], dtype=x.dtype, device="cuda")

            start_offset = 0
            for logical_width in layer.logical_widths:
                end_offset = start_offset + logical_width

                # pass w_scales directly here
                w_scales = layer.weight_scale[start_offset:end_offset]
                w_scale = w_scales.max()
                assert torch.all(w_scales == w_scale)

                # pass w_scales here
                out, _ = torch._scaled_mm(
                    qinput,
                    layer.weight[start_offset:end_offset, :].t(),
                    out_dtype=x.dtype,
                    scale_a=x_scale,
                    scale_b=w_scale,
                    bias=bias,
                )
                output[:, start_offset:end_offset] = out
                start_offset = end_offset

            return output


def per_tensor_quantize_static(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
        inv_scale: The scale.
    """
    # Scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def per_tensor_quantize_dyanmic(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor dynamic scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    min_val, max_val = tensor.aminmax()
    amax = min_val.abs().max(max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale
