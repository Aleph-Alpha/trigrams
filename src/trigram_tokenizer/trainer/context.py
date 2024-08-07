from typing import Dict, Any, Union
from pathlib import Path
import random
import numpy as np
import torch

from .config import TrainerConfig
from trigram_tokenizer.logging import logger


class TrainerContext:
    """
    Context containing information for a train or inference process
    The config is regarded to be immutable.
    """

    def __init__(
        self,
        config: TrainerConfig,
    ):
        self.config = config

        self.iterations = 0
        self.consumed_samples = 0

    def step(self):
        self.iterations += 1
        self.consumed_samples += (
            self.config.training.gradient_accumulation_steps
            * self.config.training.micro_batch_size
            * self.config.training.world_size
        )

    def state_dict(self):
        return {
            "iterations": self.iterations,
            "consumed_samples": self.consumed_samples,
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.iterations = state_dict["iterations"]
        self.consumed_samples = state_dict["consumed_samples"]
        random.setstate(state_dict["random_rng_state"])
        np.random.set_state(state_dict["np_rng_state"])
        torch.set_rng_state(state_dict["torch_rng_state"])
        torch.cuda.set_rng_state(state_dict["torch_cuda_rng_state"])

    def save(self, dir: Union[Path, str]):
        """
        Save the context state to a directory.
        Assumption is that there are no name collisions of files.
        """

        dir = Path(dir)
        if not dir.is_dir():
            dir.mkdir(parents=True)
        if torch.distributed.get_rank() == 0:
            self.config.save(dir / "config.yml")
        torch.save(
            self.state_dict(),
            str(dir / f"context_global_rank_{torch.distributed.get_rank()}.pt"),
        )

    def load_checkpoint(self, dir: Union[Path, str]):
        """
        Load the state into an already initialized context
        """
        dir = Path(dir)
        logger.info(f"loading context checkpoint from {dir}")

        # load checkpoint file if exists
        checkpoint_file = dir / f"context_global_rank_{torch.distributed.get_rank()}.pt"
        if checkpoint_file.is_file():
            state_dict = torch.load(str(checkpoint_file))
            self.load_state_dict(state_dict)

        # if the context checkpoint does not exist, new global ranks are in play
        # in this case iterations and consumed samples need to be synced
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                t = torch.tensor(
                    [
                        self.iterations,
                        self.consumed_samples,
                    ]
                ).cuda()
            else:
                t = torch.tensor([0, 0]).cuda()

            torch.distributed.all_reduce(
                t,
                op=torch.distributed.ReduceOp.MAX,
                group=torch.distributed.group.WORLD,
            )

            self.iterations = int(t[0].item())
            self.consumed_samples = int(t[1].item())

        logger.info(f"loaded context checkpoint from {dir}")
