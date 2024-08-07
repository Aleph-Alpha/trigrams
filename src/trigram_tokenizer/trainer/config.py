from typing import Optional
from pydantic import Field
from pathlib import Path

from ..logging import LoggerConfig, LogLevel
from ..config import BaseConfig
from ..transformer import TransformerArchitectureConfig, Precision, EmbeddingAggregation
from ..data import TextDatasetConfig
from ..tokenizer import TrigramTokenizerConfig
from ..optimizer import (
    AdamWOptimizerConfig,
    LearningRateSchedulerConfig,
    LossScalerConfig,
    LearningRateDecayStyle,
)


class TrainingConfig(BaseConfig):
    iterations: int = Field(description="")
    world_size: int = Field(1, description="")
    micro_batch_size: int = Field(description="")
    gradient_accumulation_steps: int = Field(description="")

    save_dir: Optional[str] = Field(None, description="")
    save_interval: int = Field(description="")
    load_dir: Optional[str] = Field(None, description="")
    load_optimizer_states: bool = Field(True, description="")
    load_context: bool = Field(True, description="")

    weight_decay: float = Field(0.1, description="")
    loss_pos_weight: float = Field(100.0, description="")
    loss_scale: float = Field(1000.0, description="")

    dataloader_num_workers: int = Field(0, description="")
    dataloader_pin_memory: bool = Field(True, description="")

    determined_experiment_id: Optional[int] = Field(None, description="")
    determined_trial_id: Optional[int] = Field(None, description="")

    seed: int = Field(42, description="")
    profile: bool = Field(False, description="")

    reset_attention_mask: bool = Field(True, description="")
    gather_word_statistics: bool = Field(False, description="")

    statistics_every_steps: Optional[int] = Field(None, description="")


class TrainerConfig(BaseConfig):
    logger: LoggerConfig = Field(
        LoggerConfig(
            log_level=LogLevel.INFO,
            log_dir=None,
            metrics_ranks=None,
            use_wandb=False,
            wandb_ranks=None,
            wandb_host="https://api.wandb.ai",
            wandb_team="aleph-alpha",
            wandb_project="trigram-tokenizer",
            wandb_group="default",
            wandb_api_key=None,
            use_tensorboard=False,
            tensorboard_ranks=None,
            determined_metrics_ranks=None,
        ),
        description="",
    )

    optimizer: AdamWOptimizerConfig = Field(
        AdamWOptimizerConfig(
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            gradient_clipping=0.0,
            allreduce_bucket_size=500000000,
            loss_scaler=LossScalerConfig(
                enable=False,
                initial_scale=2.0**32,
                window=1000,
                hysteresis=2,
                consecutive_hysteresis=False,
                min_scale=1.0,
                factor=2.0,
            ),
            zero=True,
            zero_save_static=False,
            debug_log=False,
        ),
        description="",
    )

    learning_rate_scheduler: LearningRateSchedulerConfig = Field(
        LearningRateSchedulerConfig(
            learning_rate=0.0001,
            learning_rate_minimum=0.0,
            learning_rate_decay_style=LearningRateDecayStyle.COSINE,
            learning_rate_decay_iters=50000,
            learning_rate_warmup_steps=2000,
        ),
        description="",
    )

    architecture: TransformerArchitectureConfig = Field(
        TransformerArchitectureConfig(
            vocab_size=32000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=2,
            norm="rms",
            mlp_type="simple_gelu",
            init="xavier",
            rotary_embedding_base=10000,
            sequence_length=2048,
            mlp_factor=4.0,
            precision=Precision.FLOAT32,
            layernorm_epsilon=0.00001,
            init_std_global_gain=1.0,
            use_flash=True,
            embedding_normalization=False,
            embedding_aggregation=EmbeddingAggregation.SUM,
            bias_terms=False,
        ),
        description="",
    )

    tokenizer: TrigramTokenizerConfig = Field(
        TrigramTokenizerConfig(
            lowercase=False,
            sequence_length=2048,
            vocab_size=32000,
            vocab_population=4,
            seed=42,
            end_of_text="<|endoftext|>",
            cache_dir=None,
            vocab_population_partial_lowercase=0,
            do_classic_tokenization=False,
            prefix_path_tokenizer_file="",
            initialize="hash",
            entire_words=False,
            word_edge_weight=1,
        )
    )

    data: TextDatasetConfig = Field(
        TextDatasetConfig(
            seed=42,
            sequence_length=2048,
            prefix_paths=[Path("/insert/path/here")],
            prefix_path_tokenizer_file=Path("/insert/path/here"),
            reset_position_ids=True,
            pretraining=True,
        ),
        description="",
    )

    training: TrainingConfig = Field(
        TrainingConfig(
            iterations=50000,
            world_size=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            save_interval=1,
            save_dir=None,
            load_dir=None,
            load_optimizer_states=True,
            load_context=True,
            weight_decay=0.1,
            loss_pos_weight=100.0,
            loss_scale=1000.0,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            determined_experiment_id=None,
            determined_trial_id=None,
            seed=42,
            reset_attention_mask=True,
            profile=False,
            gather_word_statistics=False,
            statistics_every_steps=None,
        )
    )
