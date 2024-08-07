from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Dict
from pathlib import Path

import torch

from ..config import BaseConfig


class OptimizerStepOutput(NamedTuple):
    global_grad_norm: Optional[float]
    global_grad_norm_clipped: Optional[float]
    learning_rates: Optional[Dict[str, float]]
    overflow: Optional[bool]
    no_overflow_steps: Optional[int]
    current_loss_scale: Optional[float]
    debug_dict: Optional[Dict[str, float]]


class BaseOptimizer(ABC):
    def __init__(self, config: BaseConfig):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def _assert_no_parameter_duplicates(self):
        """
        Make sure no parameter occurs twice within one or more parameter groups
        """
        raise NotImplementedError

    @abstractmethod
    def step(self) -> OptimizerStepOutput:
        """
        Do a model step, this optimizes model parameters and zeros gradients
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: torch.Tensor):
        """
        execute backward pass on loss and potentiall add loss scaling
        """
        raise NotImplementedError

    @abstractmethod
    def log_state(self):
        """
        Log useful information for debugging and overal information
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """
        Get a state_dict fully representing the optimizer state
        A load of such state dict fully restores the state of the optimizer.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, dir: Path):
        """
        Save the optimizer state to a directory.
        Assumption is that there are no name collisions of files.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, dir: Path):
        """
        Load the state into an already initialized optimizer
        """
        raise NotImplementedError

    @abstractmethod
    def refresh_optimizer_after_model_change(self):
        """
        Refresh the optimizer after the model has been changed.
        This may be necessary in case of tensor copies.
        """
        raise NotImplementedError
