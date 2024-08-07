from typing import Dict, Optional
from pathlib import Path
import pytest
import os
from unittest import mock

import torch

from ..utils import dist_launcher, find_free_port, get_empty_cache_dir
from ..utils_determined import make_mock_cluster_info, get_determined_context

from ..utils import get_empty_cache_dir

from trigram_tokenizer.trainer.train_determined import main as main_determined
from trigram_tokenizer.trainer import TrainerConfig, Trainer


@mock.patch("uuid.uuid4")
@mock.patch("determined.get_cluster_info")
def run_test_training_determined(
    mock_cluster_info,
    mock_uuid,
    return_dict: dict,
    config_dict: dict,
    checkpoint_dir: str,
    _world_size: int,
):
    """
    function implementing the behavior of training for one single gpu / process
    """

    cluster_info = make_mock_cluster_info(_world_size, checkpoint_dir)
    cluster_info._latest_checkpoint = os.environ.get("DET_LATEST_CHECKPOINT")
    mock_cluster_info.return_value = cluster_info
    mock_uuid.return_value = "determined_checkpoint"
    with get_determined_context(checkpoint_dir) as determined_context:
        metrics_list = main_determined(
            determined_context,
            None,
            overwrite_config=config_dict,
            return_metrics=True,
            info=cluster_info,
        )


def execute_run_training_determined(
    cache_name: str,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    precision: str,
    zero: bool,
    config_dict: Optional[Dict],
):
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    tmp_dir = get_empty_cache_dir(
        f"{cache_name}_{micro_batch_size}_{gradient_accumulation_steps}_{world_size}_{precision}_{zero}"
    )
    config = TrainerConfig.from_dict(
        config_dict
        or {
            "logger": {
                "log_level": "info",
                "log_dir": None,
                "use_wandb": False,
                "use_tensorboard": False,
            },
            "architecture": {
                "vocab_size": 32000,
                "hidden_size": 256,
                "num_layers": 2,
                "num_attention_heads": 2,
                "rotary_embedding_base": 10000,
                "sequence_length": 2048,
                "mlp_factor": 4.0,
                "precision": precision,
            },
            "data": {
                "seed": 42,
                "prefix_paths": [
                    Path(__file__).parents[1]
                    / "test_data"
                    / "data_fineweb"
                    / "CC-MAIN-2013-20"
                ],
                "pretraining": True,
                "prefix_path_tokenizer_file": str(
                    Path(__file__).parents[1]
                    / "test_data"
                    / "unigram_02pct_cc_v1.0_hf_converted_cleaned.json"
                ),
                "sequence_length": 2048,
                "reset_position_ids": True,
            },
            "tokenizer": {
                "lowercase": False,
                "vocab_size": 32000,
                "vocab_population": 4,
                "seed": 42,
                "end_of_text": "<|endoftext|>",
                "cache_dir": "tmp/tokenizer/",
                "sequence_length": 2048,
            },
            "training": {
                "iterations": 6,
                "world_size": world_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "save_dir": str(tmp_dir),
                "save_interval": 4,
                "dataloader_num_workers": 1,
                "dataloader_pin_memory": True,
                "gather_word_statistics": True,
            },
            "optimizer": {
                # AdamWOptimizerConfig
                # Base config class providing general settings for non-mutability and json serialization options
                # First coefficient used for computing running averages of gradient and its square
                "beta1": 0.9,
                # Second coefficient used for computing running averages of gradient and its square
                "beta2": 0.95,
                # term added to the denominator to improve numerical stability (default: 1e-8)
                "eps": 1.0e-15,
                # clip global l2 grads to this value, deactivate if 0.0
                "gradient_clipping": 0.0,
                # number of floating points to allreduce in one go
                "allreduce_bucket_size": 500000000,
                # Configuration of the loss scaler
                "loss_scaler": {
                    # LossScalerConfig
                    # Loss scaling is designed to combat the problem of underflowing gradients encountered at long
                    # times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
                    # scale.  Ironically, this may result in OVERflowing gradients.
                    # The optimizer then skips the update step for this particular iteration/minibatch,
                    # and the loss scaler adjusts the loss scale to a lower value.
                    # If a certain number of iterations occur without overflowing gradients detected,
                    # the loss scaler increases the loss scale once more.
                    # In this way the  loss scaler attempts to "ride the edge" of
                    # always using the highest loss scale possible without incurring overflow.
                    #
                    "enable": False,
                    # Initial loss scale
                    "initial_scale": 4294967296.0,
                    #
                    "window": 1000,
                    #
                    "hysteresis": 2,
                    #
                    "consecutive_hysteresis": False,
                    #
                    "min_scale": 1.0,
                    #
                    "factor": 2.0,
                },
                # enable zero stage 1 optimizer
                "zero": zero,
            },
            "learning_rate_scheduler": {
                # LearningRateSchedulerConfig
                # Base config class providing general settings for non-mutability and json serialization options
                # Base learning rate; this is also the maximum learning rate.
                "learning_rate": 0.007196856730011521,
                # Minimum learning rate below which a step's learning rate will never drop. This is the final learning rate after the schedule has been applied.
                "learning_rate_minimum": 0.0,
                # Shape of the learning rate decay after warm up
                "learning_rate_decay_style": "cosine",
                # Number of iterations within which the learning rate follows the schedule. Warmup iterations are included.
                "learning_rate_decay_iters": 143000,
                # Number of warmup steps during which the learning rate is linearly increased to the maximum learning rate. The actual schedule starts after the warmump steps.
                "learning_rate_warmup_steps": 200,
            },
        }
    )

    if "DET_LATEST_CHECKPOINT" in os.environ:
        del os.environ["DET_LATEST_CHECKPOINT"]

    # Train up to 10 steps
    # This function will checkpoint after 6 steps so that when called again four more steps are run
    # on the checkpoint to compare losses of a checkpoint that was trained with a resume and one that was trained continuously
    return_dict_continously_trained_model = dist_launcher(
        run_func=run_test_training_determined,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
        checkpoint_dir=tmp_dir,
        _world_size=world_size,
    )

    # Resume model training from the previous checkpoint at 6 steps.
    # Train up to 10 steps after loading from checkpoint
    # Step 6 to 10 should have the same losses for both trainings
    determined_checkpoint_dir = str(Path(tmp_dir) / "determined_checkpoint")
    os.environ["DET_LATEST_CHECKPOINT"] = str(determined_checkpoint_dir)

    return_dict_loaded_checkpoint = dist_launcher(
        run_func=run_test_training_determined,
        world_size=world_size,
        master_port=find_free_port(),
        config_dict=config.as_dict(),
        checkpoint_dir=tmp_dir,
        _world_size=world_size,
    )


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("precision", ["bfloat16"])
@pytest.mark.parametrize("zero", [True, False])
def test_training_distributed(world_size: int, precision: str, zero: bool):
    execute_run_training_determined(
        cache_name="test_training_distributed",
        world_size=world_size,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        precision=precision,
        zero=zero,
        config_dict=None,
    )


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("precision", ["bfloat16"])
@pytest.mark.parametrize("zero", [True, False])
def test_training_distributed_classic(world_size: int, precision: str, zero: bool):
    execute_run_training_determined(
        cache_name="test_training_distributed_classic",
        world_size=world_size,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        precision=precision,
        zero=zero,
        config_dict={
            "logger": {
                "log_level": "info",
                "log_dir": None,
                "use_wandb": False,
                # url of the wandb host
                "wandb_host": "https://api.wandb.ai",
                # Team name for Weights and Biases.
                "wandb_team": "aleph-alpha",
                # wandb project name
                "wandb_project": "trigram_tokenizer",
                # wandb project name
                "wandb_group": "3B",
                # set wandb api key in order not to perform a wandb login first
                "wandb_api_key": "NONE",
                "use_tensorboard": False,
            },
            "architecture": {
                "init": "pythia",
                "use_flash": True,
                "vocab_size": 128000,
                "hidden_size": 256,
                "num_layers": 2,
                "num_attention_heads": 2,
                "rotary_embedding_base": 10000,
                "sequence_length": 2048,
                "mlp_factor": 4,
                "precision": precision,
                "init_std_global_gain": 1.0,
            },
            "data": {
                "seed": 42,
                "prefix_paths": [
                    Path(__file__).parents[1]
                    / "test_data"
                    / "data_fineweb"
                    / "CC-MAIN-2013-20"
                ],
                "pretraining": True,
                "prefix_path_tokenizer_file": str(
                    Path(__file__).parents[1]
                    / "test_data"
                    / "unigram_02pct_cc_v1.0_hf_converted_cleaned.json"
                ),
                "sequence_length": 2048,
                "reset_position_ids": True,
            },
            "tokenizer": {
                "sequence_length": 2048,
                "lowercase": False,
                "initialize": "",  # "orthogonal"
                "vocab_size": 128000,
                "vocab_population": 1,
                "vocab_population_partial_lowercase": 0,  # num of vocab_population taken in lowercase
                "prefix_path_tokenizer_file": str(
                    Path(__file__).parents[1]
                    / "test_data"
                    / "unigram_02pct_cc_v1.0_hf_converted_cleaned.json"
                ),
                "do_classic_tokenization": True,  #! setting true overwrites above and uses prefix_path_tokenizer from data to do on-demand tokenization
                "seed": 42,
                "end_of_text": "<|endoftext|>",
                "cache_dir": None,  
            },
            "training": {
                "iterations": 6,
                "micro_batch_size": 2,  # 8
                "gradient_accumulation_steps": 2,  # 2
                "load_optimizer_states": True,
                "load_context": True,
                "save_interval": 4,
                "weight_decay": 0.1,
                "loss_pos_weight": 100.0,
                "loss_scale": 100.0,
                "dataloader_num_workers": 0,
                "dataloader_pin_memory": True,
                "seed": 42,
                "profile": False,
                "reset_attention_mask": True,
                "gather_word_statistics": True,
            },
            "optimizer": {
                # AdamWOptimizerConfig
                # Base config class providing general settings for non-mutability and json serialization options
                # First coefficient used for computing running averages of gradient and its square
                "beta1": 0.9,
                # Second coefficient used for computing running averages of gradient and its square
                "beta2": 0.95,
                # term added to the denominator to improve numerical stability (default: 1e-8)
                "eps": 1.0e-8,
                # clip global l2 grads to this value, deactivate if 0.0
                "gradient_clipping": 1.0,
                # number of floating points to allreduce in one go
                "allreduce_bucket_size": 500000000,
                # Configuration of the loss scaler
                "loss_scaler": {
                    # LossScalerConfig
                    # Loss scaling is designed to combat the problem of underflowing gradients encountered at long
                    # times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
                    # scale.  Ironically, this may result in OVERflowing gradients.
                    # The optimizer then skips the update step for this particular iteration/minibatch,
                    # and the loss scaler adjusts the loss scale to a lower value.
                    # If a certain number of iterations occur without overflowing gradients detected,
                    # the loss scaler increases the loss scale once more.
                    # In this way the  loss scaler attempts to "ride the edge" of
                    # always using the highest loss scale possible without incurring overflow.
                    #
                    "enable": False,
                    # Initial loss scale
                    "initial_scale": 4294967296.0,
                    #
                    "window": 1000,
                    #
                    "hysteresis": 2,
                    #
                    "consecutive_hysteresis": False,
                    #
                    "min_scale": 1.0,
                    #
                    "factor": 2.0,
                },
                # enable zero stage 1 optimizer
                "zero": zero,
            },
            #
            "learning_rate_scheduler": {
                "learning_rate": 4.5e-4,
                # Minimum learning rate below which a step's learning rate will never drop. This is the final learning rate after the schedule has been applied.
                "learning_rate_minimum": 4.5e-5,
                # Shape of the learning rate decay after warm up
                "learning_rate_decay_style": "cosine",
                # Number of iterations within which the learning rate follows the schedule. Warmup iterations are included.
                "learning_rate_decay_iters": 72_000,
                # Number of warmup steps during which the learning rate is linearly increased to the maximum learning rate. The actual schedule starts after the warmump steps.
                "learning_rate_warmup_steps": 500,
            },
        },
    )
