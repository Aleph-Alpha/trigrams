from enum import Enum
import math
from pydantic import Field
from ...config import BaseConfig


class LearningRateDecayStyle(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class LearningRateSchedulerConfig(BaseConfig):
    learning_rate: float = Field(
        0.0,
        description="Base learning rate; this is also the maximum learning rate.",
    )

    learning_rate_minimum: float = Field(
        0.0,
        description="Minimum learning rate below which a step's learning rate will never drop. This is the final learning rate after the schedule has been applied.",
    )

    learning_rate_decay_style: LearningRateDecayStyle = Field(
        LearningRateDecayStyle.COSINE,
        description="Shape of the learning rate decay after warm up",
    )

    learning_rate_decay_iters: int = Field(
        0,
        description="Number of iterations within which the learning rate follows the schedule. Warmup iterations are included.",
    )

    learning_rate_warmup_steps: int = Field(
        0,
        description="Number of warmup steps during which the learning rate is linearly increased to the maximum learning rate. The actual schedule starts after the warmump steps.",
    )


class LearningRateScheduler:
    """
    Class managing learning rate decay functions from:
    https://openreview.net/pdf?id=BJYwwY9ll pg. 4
    """

    def __init__(self, config: LearningRateSchedulerConfig):
        self.config = config

    def get_lr(self, step_index: int) -> float:
        """
        Compute the learning rate for a given step index.
        """
        # Use linear warmup for the initial part.
        if (
            self.config.learning_rate_warmup_steps > 0
            and step_index <= self.config.learning_rate_warmup_steps
        ):
            return (
                self.config.learning_rate
                * float(step_index)
                / float(self.config.learning_rate_warmup_steps)
            )

        # If constant learning rate return the max after warmup
        if self.config.learning_rate_decay_style == LearningRateDecayStyle.CONSTANT:
            return self.config.learning_rate

        # For any steps larger than `self.config.learning_rate_decay_iters`, use `self.min_lr`
        if step_index > self.config.learning_rate_decay_iters:
            return self.config.learning_rate_minimum

        # Use decay styles after warmup
        # Note that to get here self.config.learning_rate_warmup_steps < step_index <= self.config.learning_rate_decay_iters
        num_steps_nowarmup = step_index - self.config.learning_rate_warmup_steps
        decay_steps_nowarmup = (
            self.config.learning_rate_decay_iters
            - self.config.learning_rate_warmup_steps
        )
        decay_ratio = float(num_steps_nowarmup) / float(decay_steps_nowarmup)

        assert 0.0 <= decay_ratio <= 1.0
        delta_lr = self.config.learning_rate - self.config.learning_rate_minimum
        if self.config.learning_rate_decay_style == LearningRateDecayStyle.LINEAR:
            coeff = 1.0 - decay_ratio
        elif self.config.learning_rate_decay_style == LearningRateDecayStyle.COSINE:
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.config.learning_rate_minimum + coeff * delta_lr
