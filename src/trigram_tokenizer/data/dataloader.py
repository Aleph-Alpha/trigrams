from functools import partial
from typing import List
from einops import rearrange
import torch
from trigram_tokenizer.logging import logger
from .dataset import TextDataset
from trigram_tokenizer.tokenizer import EncodingBatchTraining, EncodingTraining

from multiprocessing import Semaphore


class RandomSampler:
    """
    Class with iterator that returns a list of indices for a micro batch.
    """

    def __init__(
        self,
        dataset: TextDataset,
        seed: int,
        consumed_samples: int,
        micro_batch_size: int,
        world_size: int,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.seed = seed
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.world_size = world_size

        # derive parameters
        self.total_samples = len(dataset)
        self.total_micro_batches = len(self.dataset) // self.micro_batch_size
        self.total_micro_batches_per_data_parallel = (
            self.total_micro_batches // self.world_size
        )
        self.usable_total_samples = (
            self.total_micro_batches_per_data_parallel
            * self.micro_batch_size
            * self.world_size
        )
        assert (
            self.usable_total_samples > 0
        ), "not usable samples; this means that the dataset is too small for the provided data parallel size and micro batch size"

    def __len__(self):
        return self.total_micro_batches

    def __iter__(self):
        epoch = self.consumed_samples // self.usable_total_samples
        consumed_samples_in_current_epoch = (
            self.consumed_samples % self.usable_total_samples
        )
        remaining_samples_in_current_epoch = (
            self.usable_total_samples - consumed_samples_in_current_epoch
        )

        logger.info(f"creating new dataset shuffle index for epoch {epoch}")
        logger.info(f"total_samples {self.total_samples}")
        logger.info(f"micro_batch_size {self.micro_batch_size}")
        logger.info(f"usable_total_samples {self.usable_total_samples}")
        logger.info(
            f"consumed_samples_in_current_epoch {consumed_samples_in_current_epoch}"
        )
        logger.info(
            f"remaining_samples_in_current_epoch {remaining_samples_in_current_epoch}"
        )

        # Set seed if not set already
        self.dataset.set_seed(seed=self.seed + epoch)

        idx_range = (
            (
                torch.arange(
                    0,
                    (remaining_samples_in_current_epoch // self.world_size),
                    dtype=torch.long,
                )
                * self.world_size
            )
            + torch.distributed.get_rank()
            + consumed_samples_in_current_epoch
        )

        idx_range = idx_range.tolist()
        if len(idx_range) % self.micro_batch_size != 0:
            new_len = len(idx_range) - (len(idx_range) % self.micro_batch_size)
            idx_range = idx_range[:new_len]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_size * self.world_size
                yield batch
                batch = []


def collate(
    x: List[EncodingTraining],
    use_flash=True,
    semaphore=None,
    reset_attention_mask: bool = True,
):
    result = EncodingBatchTraining.from_encodings(
        x, reset_attention_mask=reset_attention_mask
    )
    if use_flash:
        seq_len = result.attention_mask.shape[2]
        batch_size = result.attention_mask.shape[0]

        attention_mask = rearrange(result.attention_mask, "b s n h -> (b s) n h").to(
            torch.int32
        )
        attention_mask = (1 - attention_mask).sum(dim=2)

        final_attention_mask = []
        for cur in (1 == attention_mask).nonzero():
            batch, index = cur
            final_attention_mask.append((batch * seq_len + index).item())

        if final_attention_mask[-1] != batch_size * seq_len:
            final_attention_mask.append(batch_size * seq_len)

        final_attention_mask_tensor = (
            torch.tensor(final_attention_mask)
            .to(torch.int32)
            .to(result.attention_mask.device)
        )
        result.attention_mask = final_attention_mask_tensor

    if semaphore:
        semaphore.acquire()
    return result


class DataLoader(
    torch.utils.data.DataLoader
):  # inherit and have another instance as a child to guarantee the interface of the DataLoader and instantiate from our parameters to initialize the data parallel sampler
    """
    Generic base class to iterate over any given dataset which implements BaseDataset.

    The data loader
        - is instantiated from a seed, the number of consumed samples, a micro batch size and a dataset
        - implements an infinite iterator over the dataset
    """

    def __init__(
        self,
        seed: int,
        consumed_samples: int,
        dataset: TextDataset,
        micro_batch_size: int,
        world_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        use_flash: bool = True,
        reset_attention_mask: bool = True,
    ):
        """
        seed (`int`)
            seed used to shuffle the dataset
        consumed_samples (`int`)
            number of samples already consumed during training from the dataset

        dataset (`BaseDataset`)
            dataset which implements the BaseDataset interface
        """
        self.use_flash = use_flash
        self.seed = seed
        self.consumed_samples = consumed_samples
        self.dataset = dataset
        self.micro_batch_size = micro_batch_size
        self.world_size = world_size

        assert (
            len(self.dataset) >= self.micro_batch_size
        ), f"cannot instantiate data loader with micro_batch_size { self.micro_batch_size } because dataset has only length {len(self.dataset)}"

        batch_sampler = RandomSampler(
            dataset=self.dataset,
            seed=self.seed,
            consumed_samples=self.consumed_samples,
            micro_batch_size=self.micro_batch_size,
            world_size=self.world_size,
        )

        self.semaphore = Semaphore(10)

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=partial(
                collate,
                use_flash=self.use_flash,
                semaphore=self.semaphore,
                reset_attention_mask=reset_attention_mask,
            ),
            pin_memory=pin_memory,
            prefetch_factor=None,
        )

        self.iterator = self._iterate()

    def _iterate(self):
        while True:
            for item in self.dataloader:
                self.semaphore.release()
                yield item

    def __next__(self):
        return next(self.iterator)
