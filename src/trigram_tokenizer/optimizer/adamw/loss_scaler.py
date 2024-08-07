from typing import Dict, Any, List, NamedTuple, Optional
from pydantic import Field

import torch

from trigram_tokenizer.logging import logger

from .adamw_parameter_group import AdamWParameterGroup
from ...config import BaseConfig


def has_inf_or_nan(x: torch.Tensor):
    try:
        # if x is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as x
        # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # cpu_sum = float(x.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float("inf") or cpu_sum == -float("inf") or cpu_sum != cpu_sum:
            return True
        return False


class LossScalerConfig(BaseConfig):
    """
    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.

    The optimizer then skips the update step for this particular iteration/minibatch,
    and the loss scaler adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    the loss scaler increases the loss scale once more.
    In this way the  loss scaler attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.
    """

    enable: bool = Field(
        False,
        description="",
    )

    initial_scale: float = Field(
        2.0**32,
        description="Initial loss scale",
    )

    window: int = Field(
        1000,
        description="",
    )

    hysteresis: float = Field(
        2,
        description="",
    )

    consecutive_hysteresis: bool = Field(
        False,
        description="",
    )

    min_scale: float = Field(
        1.0,
        description="",
    )

    factor: float = Field(
        2.0,
        description="",
    )


class LossScalerOutput(NamedTuple):
    overflow: Optional[bool]
    no_overflow_steps: Optional[int]
    current_loss_scale: Optional[float]


class LossScaler:
    def __init__(
        self, config: LossScalerConfig, parameter_groups: List[AdamWParameterGroup]
    ):
        self.config = config

        self._current_scale = self.config.initial_scale
        self._current_hysteresis = self.config.hysteresis
        self._no_overflow_steps = 0

        # record parameters for overflow checking
        parameters = list()
        for parameter_group in parameter_groups:
            parameters.extend(parameter_group.parameters_for_overflow_check)
        self.parameters = parameters

    def some_overflow_in_local_param_grads(self):
        for parameter in self.parameters:
            if parameter.grad is not None and has_inf_or_nan(parameter.grad.data):
                return True
        return False

    def some_overflow_in_global_param_grads(self):
        local_overflow = self.some_overflow_in_local_param_grads()
        overflow_tensor = torch.cuda.ByteTensor([local_overflow])

        torch.distributed.all_reduce(
            overflow_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=torch.distributed.group.WORLD,
        )

        overflow = overflow_tensor[0].item()
        return bool(overflow)

    def step(self):
        if not self.config.enable:
            return LossScalerOutput(
                overflow=None, no_overflow_steps=None, current_loss_scale=None
            )

        # check overflow
        overflow = self.some_overflow_in_global_param_grads()

        # apply loss scaling dependent on overflow
        if overflow:
            if self.config.hysteresis == 1 or self._current_hysteresis == 1:
                self._current_scale = max(
                    self._current_scale / self.config.factor,
                    self.config.min_scale,
                )
            else:
                self._current_hysteresis -= 1
            self._no_overflow_steps = 0
        else:
            if self.config.consecutive_hysteresis:
                self._current_hysteresis = self.config.hysteresis
            if (
                self._no_overflow_steps > 0
                and (self._no_overflow_steps) % self.config.window == 0
            ):
                if not self.config.consecutive_hysteresis:
                    self._current_hysteresis = self.config.hysteresis
                self._current_scale *= self.config.factor
            self._no_overflow_steps += 1

        return LossScalerOutput(
            overflow=overflow,
            no_overflow_steps=self._no_overflow_steps,
            current_loss_scale=self._current_scale,
        )

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.config.enable:
            return loss

        loss = loss * self._current_scale
        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_scale": self._current_scale,
            "current_hysteresis": self._current_hysteresis,
            "no_overflow_steps": self._no_overflow_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._current_scale = state_dict["current_scale"]
        self._current_hysteresis = state_dict["current_hysteresis"]
        self._no_overflow_steps = state_dict["no_overflow_steps"]
