from typing import Optional, Union, Dict
from pathlib import Path
import time
import collections
import uuid
import sys


import torch
from time import gmtime, strftime
import gc

from ..tokenizer import EncodingBatchTraining
from .context import TrainerContext
from trigram_tokenizer.logging import logger
from ..transformer import TransformerLMHeadModel
from ..data import TextDataset, DataLoader
from ..optimizer import AdamWOptimizer
from pathlib import Path


@torch.jit.script
def torch_move_stuff(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: torch.Tensor,
    vocab_size: int,
):
    logits = logits.float()
    targets = targets.to(logits.device).float()
    targets[
        targets > 1.0
    ] = 1.0  # binarize targets, differnt loss functions could be tested if we have collisions

    loss_weights_repeated = (loss_weights.to(logits.device)[:, :, None]).repeat(
        1, 1, vocab_size
    )

    return logits, targets, loss_weights_repeated


class Trainer:
    def __init__(
        self,
        context: TrainerContext,
        transformer: TransformerLMHeadModel,
        optimizer: AdamWOptimizer,
        dataset: TextDataset,
        device: Optional[Union[str, torch.device]],
    ):
        self.context = context

        self.transformer = transformer
        self.optimizer = optimizer
        self.dataset = dataset
        self.device = device
        self.words_target: collections.Counter = collections.Counter()

        self.load_checkpoint()
        self.accuracy_loss_occurance_dict: Dict[str, int] = {}

        print(
            f" >> DATALOADER: {self.context.config.training.dataloader_num_workers} {self.context.config.training.world_size} {self.context.config.training.dataloader_pin_memory}"
        )
        if not self.context.config.data.pretraining:
            assert not self.context.config.training.reset_attention_mask
            assert not self.context.config.data.reset_position_ids

        self.dataloader: DataLoader = DataLoader(
            use_flash=self.context.config.architecture.use_flash,
            seed=self.context.config.training.seed,
            consumed_samples=self.context.consumed_samples,
            dataset=self.dataset,
            micro_batch_size=self.context.config.training.micro_batch_size,
            world_size=self.context.config.training.world_size,
            num_workers=self.context.config.training.dataloader_num_workers,
            pin_memory=self.context.config.training.dataloader_pin_memory,
            reset_attention_mask=self.context.config.training.reset_attention_mask,
        )
        gc.collect()
        torch.cuda.empty_cache()

    def run_training(self):
        logger.info(f"running training")

        if self.context.config.training.profile:
            formatted_time = strftime("%m-%d_%H_%M_%S", gmtime())
            profile_path = (
                f"./logs/profile_{formatted_time}"
            )
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    skip_first=0, wait=0, warmup=1, active=3, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()
        count = 0
        while self.context.iterations < self.context.config.training.iterations:
            count += 1
            if count % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            if self.context.config.training.profile:
                prof.step()
            self.train_step()

            if logger._use_determined:
                assert logger.determined_context is not None
                if logger.determined_context.preempt.should_preempt():
                    self.determined_save_checkpoint()
                    print("exiting program after preemption.", flush=True)
                    sys.exit()

            # save checkpoint
            if (
                self.context.iterations % self.context.config.training.save_interval
                == 0
            ):
                if logger._use_determined:
                    self.determined_save_checkpoint()
                else:
                    self.save()

    def train_step(self):
        start = time.time()
        self.optimizer.zero_grad()
        step_metrics = list()

        for _gradient_accumulation_step in range(
            self.context.config.training.gradient_accumulation_steps
        ):
            batch = next(self.dataloader)

            assert isinstance(batch, EncodingBatchTraining)

            if self.context.config.training.gather_word_statistics:
                for words_target in batch.words_targets:
                    for word_target in words_target:
                        if word_target is None:
                            continue
                        self.words_target[word_target] += 1

            logits = self.transformer(
                trigram_set_position_ids=batch.trigram_set_position_ids.detach().to(
                    self.device
                ),
                trigram_token_ids=batch.trigram_token_ids.detach().to(self.device),
                trigram_token_ids_offsets=batch.trigram_token_ids_offsets.detach().to(
                    self.device
                ),
                position_ids=batch.position_ids.detach().to(self.device).to(torch.long),
                attention_mask=batch.attention_mask.detach().to(self.device),
            )
            loss, metrics = self.loss_fn(
                logits=logits,
                targets=batch.targets,
                loss_weights=batch.loss_weights,
                do_metrics=self.context.config.training.statistics_every_steps
                is not None
                and (
                    (1 + self.context.iterations)
                    % self.context.config.training.statistics_every_steps
                    == 0
                ),
            )

            step_metrics.append(metrics)
            self.optimizer.backward(
                loss
            )  # loss returned is only used for the backward pass, all for logging is in metrics

        # optimize
        optimizer_step_output = self.optimizer.step()

        # context
        self.context.step()

        end = time.time()

        # aggregate metrics and log
        aggregated_metrics = {}
        values_list = list()
        for k in sorted(list(step_metrics[0].keys())):
            values_list.append(
                0.0
                if any(m[k] is None for m in step_metrics)
                else torch.tensor(
                    [m[k] for m in step_metrics],
                    dtype=torch.float32,
                    device=self.device,
                ).mean()
            )

        values_tensor = torch.tensor(values_list, dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(values_tensor)
        values_tensor = values_tensor / self.context.config.training.world_size

        for k, v in zip(
            sorted(list(step_metrics[0].keys())), values_tensor.cpu().tolist()
        ):
            aggregated_metrics[k] = v

        # overall
        aggregated_metrics["runtime/step_duration"] = float(end - start)
        aggregated_metrics["training/step"] = self.context.iterations
        aggregated_metrics["training/epochs"] = self.context.consumed_samples / len(
            self.dataset
        )

        # optim
        aggregated_metrics[
            "training/global_grad_norm"
        ] = optimizer_step_output.global_grad_norm
        if optimizer_step_output.debug_dict is not None:
            aggregated_metrics.update(optimizer_step_output.debug_dict)
        if optimizer_step_output.learning_rates is not None:
            for (
                param_group_name,
                learning_rate,
            ) in optimizer_step_output.learning_rates.items():
                aggregated_metrics[
                    f"training/learning_rate_{param_group_name}"
                ] = learning_rate
        if optimizer_step_output.overflow is not None:
            aggregated_metrics["training/overflow"] = int(
                optimizer_step_output.overflow
            )
        if optimizer_step_output.no_overflow_steps is not None:
            aggregated_metrics[
                "training/no_overflow_steps"
            ] = optimizer_step_output.no_overflow_steps
        if optimizer_step_output.current_loss_scale is not None:
            aggregated_metrics[
                "training/current_loss_scale"
            ] = optimizer_step_output.current_loss_scale

        logger.log_metrics(aggregated_metrics, self.context.iterations)

    def loss_fn(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: torch.Tensor,
        do_metrics: bool,
    ):
        if self.context.config.tokenizer.do_classic_tokenization:
            logits = logits.float()
            targets = targets.to(logits.device).squeeze().reshape(-1).to(torch.int64)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets
            )
            metrics = {"training/loss": float(loss.item())}
        else:
            logits, targets, loss_weights_repeated = torch_move_stuff(
                logits, targets, loss_weights, self.dataset.tokenizer.config.vocab_size
            )

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
                pos_weight=torch.tensor(
                    self.context.config.training.loss_pos_weight,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
            )
            loss = loss * loss_weights_repeated.to(loss.dtype)
            loss = self.context.config.training.loss_scale * loss.mean()
            metrics = {
                "training/loss": float(loss.item())
                / self.context.config.training.loss_scale
            }
            if do_metrics:
                logits_activated = logits.sigmoid()

                logits_activated = logits_activated.view(-1)[
                    loss_weights_repeated.reshape(-1) > 0.0
                ]

                metrics = {
                    "training/loss": float(loss.item())
                    / self.context.config.training.loss_scale,
                    "charts/average_score": float(logits_activated.mean().item()),
                }
                non_padded_sequence_count = float(loss_weights.sum().item())
                for i in range(11):
                    cutoff_i = i / 10
                    metrics[f"charts/count_active_{cutoff_i}"] = (
                        float(
                            (logits_activated > cutoff_i).sum(-1).float().mean().item()
                        )
                        / non_padded_sequence_count
                    )

        return loss, metrics

    def determined_save_checkpoint(self):
        from determined.common import storage  # type: ignore

        determined_context = logger.determined_context
        storage_manager = determined_context.checkpoint._storage_manager
        if torch.distributed.get_rank() == 0:
            # Only run this once
            metadata = {
                "steps_completed": self.context.iterations,
            }
            storage_id = str(uuid.uuid4())
            with storage_manager.store_path(storage_id) as path:
                # Broadcast checkpoint path to all ranks.
                determined_context.distributed.broadcast((storage_id, path))

                self.save(save_dir=path)

                # If the storage manager is a sharedfs, then the checkpoint directory
                # will already contain all the files.  Otherwise, checkpoint files are
                # saved to a local directory before being uploaded to cloud storage so
                # we'll need to gather all the files across nodes before reporting the
                # checkpoint.
                resources = storage.StorageManager._list_directory(path)
                if isinstance(storage_manager, storage.SharedFSStorageManager):
                    all_resources = [resources]
                else:
                    # Gather resources across nodes.
                    all_resources = determined_context.distributed.gather(resources)
            resources = {k: v for d in all_resources for k, v in d.items()}

            determined_context.checkpoint._report_checkpoint(
                storage_id, resources, metadata
            )

        else:
            storage_id, path = determined_context.distributed.broadcast(None)
            self.save(save_dir=path)
            if not isinstance(storage_manager, storage.SharedFSStorageManager):
                # Gather resources across nodes.
                if determined_context.distributed.local_rank == 0:
                    resources = storage.StorageManager._list_directory(path)
                else:
                    resources = {}
                _ = determined_context.distributed.gather(resources)
            if determined_context.distributed.local_rank == 0:
                storage_manager.post_store_path(str(path), storage_id)

    def save(self, save_dir: Optional[Union[Path, str]] = None):
        iteration_str = str(self.context.iterations).zfill(12)

        if save_dir is None:
            if self.context.config.training.save_dir is None:
                return
            checkpoint_dir = Path(self.context.config.training.save_dir) / iteration_str
        else:
            checkpoint_dir = Path(save_dir) / iteration_str

        checkpoint_dir_words = checkpoint_dir / "words_target"
        checkpoint_dir_optimizer = checkpoint_dir / "optimizer"

        self.dataset.close_all()
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.mkdir(parents=True)
            self.context.save(checkpoint_dir / "context")
            self.transformer.save(checkpoint_dir / "transformer")
            self.dataset.tokenizer.save(checkpoint_dir / "tokenizer")

            checkpoint_dir_words.mkdir(parents=True)
            checkpoint_dir_optimizer.mkdir(parents=True)

        torch.distributed.barrier()
        self.optimizer.save_checkpoint(checkpoint_dir_optimizer)
        torch.save(
            self.words_target,
            checkpoint_dir_words
            / f"words_target_rank_{torch.distributed.get_rank()}.pt",
        )
        logger.info(f"saved words to {checkpoint_dir_words}")

        if torch.distributed.get_rank() == 0:
            with open(checkpoint_dir.parent / "latest", "w", encoding="UTF-8") as f:
                f.write(iteration_str)
        torch.distributed.barrier()

        logger.info(f"saved trainer to {checkpoint_dir}")
        self.dataset.open_all()

    def load_checkpoint(self):
        load_dir = self.context.config.training.load_dir
        continue_det_experiment = False

        #  Check if a determined latest checkpoint is available for example through pausing and resuming of an experiment
        if logger._use_determined:
            import determined as det  # type: ignore

            info = det.get_cluster_info()
            if info is not None and info.latest_checkpoint is not None:
                continue_det_experiment = True
                assert logger.determined_context is not None
                with logger.determined_context.checkpoint.restore_path(
                    info.latest_checkpoint
                ) as load_path:
                    logger.info(
                        f"Updating load checkpoint directory from {self.context.config.training.load_dir} to {load_path} according to determined setting"
                    )
                    load_dir = Path(load_path)
            else:
                # No latest checkpoint available from determined
                # We could still have configured a checkpoint to load and finetune
                pass

        # Check if a checkpoint load directory is specified
        if load_dir is None:
            return False

        load_dir = Path(load_dir)

        if (load_dir / "latest").is_file():
            with open(load_dir / "latest", "r", encoding="UTF-8") as f:
                global_step_dir = (
                    f.read().strip()  # strip removes potential line breaks and spaces
                )

            iteration_dir = load_dir / global_step_dir

        elif len(list((load_dir.glob("*.pt")))) > 0:
            logger.info(
                f"no latest file found, using load dir directly instead: {load_dir}"
            )
            iteration_dir = load_dir
        else:
            logger.error(f"no files found in load dir: {load_dir}")
            return False

        if not iteration_dir.is_dir():
            logger.error(f"iteration_dir does not exist: {iteration_dir}")
            return False

        if self.context.config.training.load_context or continue_det_experiment:
            self.context.load_checkpoint(iteration_dir / "context")

        self.transformer.load_checkpoint(iteration_dir / "transformer")

        if (
            self.context.config.training.load_optimizer_states
            or continue_det_experiment
        ):
            self.optimizer.load_checkpoint(iteration_dir / "optimizer")
        else:
            self.optimizer.refresh_optimizer_after_model_change()

        word_files_with_rank_for_load = [
            (str(f), int(f.stem.split("_")[-1]) % torch.distributed.get_world_size())
            for f in (iteration_dir / "words_target").glob("*.pt")
        ]

        for word_file, rank_for_load in word_files_with_rank_for_load:
            if rank_for_load == torch.distributed.get_rank():
                words_target = torch.load(word_file)
                assert isinstance(words_target, collections.Counter)
                self.words_target += words_target
                logger.info(f"loaded words_target: {word_file}")

        # tokenizer load checkpoint
        logger.info(f"loaded checkpoint: {iteration_dir}")

        return True
