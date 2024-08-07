import torch
from enum import Enum
from pydantic import Field

from ...config import BaseConfig


class Precision(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

    @property
    def dtype(self) -> torch.dtype:
        if self == Precision.FLOAT16:
            return torch.float16
        elif self == Precision.BFLOAT16:
            return torch.bfloat16
        elif self == Precision.FLOAT32:
            return torch.float32
        else:
            raise NotImplementedError


class EmbeddingAggregation(Enum):
    SUM = "sum"
    MEAN = "mean"


class TransformerArchitectureConfig(BaseConfig):
    """
    Transformer architecture config object containing non-mutable (constant) architecture specific configurations
    """

    use_flash: bool = Field(True)

    embedding_normalization: bool = Field(
        False,
        description="",
    )

    embedding_aggregation: EmbeddingAggregation = Field(
        EmbeddingAggregation.MEAN, description=""
    )

    norm: str = Field("rms")
    mlp_type: str = Field("simple_gelu")
    init: str = Field("xavier")
    bias_terms: bool = Field(True)

    vocab_size: int = Field(
        0,
        description="Size of the vocabulary before padding; this matches the vocab size of the tokenizer",
    )

    hidden_size: int = Field(
        0,
        description="Transformer hidden size.",
    )

    num_layers: int = Field(
        0,
        description="Number of luminous layers",
    )

    num_attention_heads: int = Field(
        0,
        description="Number of attention heads",
    )

    rotary_embedding_base: int = Field(
        10000,
        description="",
    )

    sequence_length: int = Field(
        2048,
        description="Sequence length in number of tokens in one sample on which a train job is run; at inference time the seqence length of a sample should (usually) not be exceeded.",
    )

    mlp_factor: float = Field(
        4.0,
        description="expansion factor for mlp hidden layer",
    )

    precision: Precision = Field(Precision.FLOAT32, description="")

    layernorm_epsilon: float = Field(
        0.00001,
        description="",
    )

    init_std_global_gain: float = Field(
        1.0,
        description="",
    )
