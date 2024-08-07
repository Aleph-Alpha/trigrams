from enum import Enum
from typing import Optional, List, Union
from pydantic import Field
import torch

try:
    from flash_attn.ops.rms_norm import rms_norm as flash_attn_rms_norm
except ImportError:
    flash_attn_rms_norm = None


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dimensions: int,
        device: Optional[Union[torch.device, str]],
        eps: float,
        dtype=torch.float32,
    ):
        super().__init__()
        self.eps = eps

        self.weight = torch.nn.Parameter(
            torch.ones(dimensions, dtype=dtype, device=device)
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if flash_attn_rms_norm is None:
            output = self._norm(x.float()).type_as(x)
            output = output * self.weight
        else:
            output = flash_attn_rms_norm(x, self.weight, self.eps)

        return output
