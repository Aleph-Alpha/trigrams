from typing import Optional, Union
import math

import torch

from ..config import TransformerArchitectureConfig, EmbeddingAggregation


class Embedding(torch.nn.Module):
    def __init__(
        self,
        config: TransformerArchitectureConfig,
        device: Optional[Union[str, torch.device]],
    ):
        super().__init__()
        self.config = config

        self.embedding = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.precision.dtype,
            device=device,
        )

        if self.config.init == "xavier":
            torch.nn.init.xavier_normal_(self.embedding.weight)
        elif self.config.init == "pythia":
            std = math.sqrt(2 / (5 * config.hidden_size))
            torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        else:
            assert self.config.init == "normal"
            torch.nn.init.normal_(
                self.embedding.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )

    def forward(
        self,
        trigram_set_position_ids: torch.Tensor,
        trigram_token_ids: torch.Tensor,
        trigram_token_ids_offsets: torch.Tensor,
        seq_len: int,
    ):
        activations_ = list()
        for start, end in trigram_token_ids_offsets.tolist():
            assert start < end

            assert (
                trigram_set_position_ids[end - 1] < self.config.sequence_length
            )  # might get padded / will get 0 value + attention mask
            assert trigram_set_position_ids[start] == 0

            # embed token ids
            embeddings = self.embedding(trigram_token_ids[start:end])

            # aggregate sparse representation
            if self.config.embedding_aggregation == EmbeddingAggregation.MEAN:
                embeddings_ = embeddings.mean(-2)
            elif self.config.embedding_aggregation == EmbeddingAggregation.SUM:
                embeddings_ = embeddings.sum(-2)
            else:
                raise NotImplementedError

            activation_ = torch.zeros(
                (seq_len, embeddings_.shape[-1]),
                dtype=embeddings_.dtype,
                device=embeddings_.device,
            )

            activation_.index_add_(
                0,
                trigram_set_position_ids[start:end].view(-1),
                embeddings_.view(-1, embeddings_.shape[-1]),
            )

            if (
                self.config.embedding_normalization
            ):  # divide by num trigrams -> mean embedding per word, not accumulated
                # this is a tiny bit wrong , but othw i get division by zero :'(
                trigram_counts = torch.zeros(
                    (seq_len),
                    dtype=torch.int32,
                    device=embeddings_.device,
                )

                trigram_counts.index_add_(
                    0,
                    trigram_set_position_ids[start:end].view(-1),
                    torch.ones_like(trigram_set_position_ids[start:end]),
                )
                trigram_counts[trigram_counts == 0] = 1.0

                activation_ = activation_ / trigram_counts[:, None]

            activations_.append(activation_)

        activations = torch.stack(activations_)
        return activations
