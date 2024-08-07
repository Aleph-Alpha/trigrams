from typing import Optional, Union, Optional, Dict, Tuple
from pathlib import Path
import math

import torch

from trigram_tokenizer.logging import logger
from .config import TransformerArchitectureConfig
from .components import Embedding, Layer, MLP

from .components.rmsnorm import RMSNorm


class TransformerModel(torch.nn.Module):
    """
    Main transformer decoder class
    """

    def __init__(
        self,
        config: TransformerArchitectureConfig,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        self.config = config

        # embedding input layer
        self.embeddings = Embedding(config=config, device=device)

        # transformer layers
        layers_ = list()
        for layer_index in range(config.num_layers):
            layer = Layer(config=config, layer_index=layer_index, device=device)
            layers_.append(layer)
        self.layers = torch.nn.ModuleList(layers_)

        # final norm
        self.norm: Union[torch.nn.LayerNorm, RMSNorm]
        if self.config.norm == "layer":
            self.norm = torch.nn.LayerNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                dtype=config.precision.dtype,
                device=device,
            )
        else:
            assert self.config.norm == "rms"
            self.norm = RMSNorm(
                dimensions=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                dtype=config.precision.dtype,
                device=device,
            )

    def forward(
        self,
        trigram_set_position_ids: torch.Tensor,
        trigram_token_ids: torch.Tensor,
        trigram_token_ids_offsets: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        # embed
        # breakpoint()

        hidden_states = self.embeddings(
            trigram_set_position_ids=trigram_set_position_ids,
            trigram_token_ids=trigram_token_ids,
            trigram_token_ids_offsets=trigram_token_ids_offsets,
            seq_len=position_ids.shape[-1],  # self.config.sequence_length,
        )

        # transformer layers
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # final norm
        norm_out = self.norm(
            hidden_states.to(self.norm.weight.device),
        )

        return norm_out, kv_cache


class TransformerLMHeadModel(torch.nn.Module):
    """
    Main transformer decoder class with lm head
    """

    def __init__(
        self,
        config: TransformerArchitectureConfig,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        self.config = config

        self.transformer = TransformerModel(config=config, device=device)

        # lm head
        self.lm_head = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.precision.dtype,
            device=device,
        )
        if self.config.init == "xavier":
            torch.nn.init.xavier_normal_(self.lm_head.weight)
        elif self.config.init == "pythia":
            std = math.sqrt(2 / (5 * config.hidden_size))
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        else:
            assert self.config.init == "normal"

            torch.nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )

    def forward(
        self,
        trigram_set_position_ids: torch.Tensor,
        trigram_token_ids: torch.Tensor,
        trigram_token_ids_offsets: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        hidden_states, kv_cache = self.transformer(
            trigram_set_position_ids=trigram_set_position_ids,
            trigram_token_ids=trigram_token_ids,
            trigram_token_ids_offsets=trigram_token_ids_offsets,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(hidden_states)

        if kv_cache is None:
            return logits
        else:
            return logits, kv_cache

    def save(self, dirname: Path):
        dirname = Path(dirname)
        dirname.mkdir(parents=True)

        torch.save(self.state_dict(), str(dirname / "transformer.pt"))
        self.config.save(dirname / "transformer_config.yaml")

    @classmethod
    def load(
        cls,
        dirname: Path,
        device: Optional[Union[str, torch.device]] = None,
        overwrite_values=None,
    ):
        dirname = Path(dirname)
        config = TransformerArchitectureConfig.from_yaml(
            dirname / "transformer_config.yaml", overwrite_values=overwrite_values
        )
        transformer = cls(config=config, device=device)
        state_dict = torch.load(str(dirname / "transformer.pt"), map_location="cpu")
        transformer.load_state_dict(state_dict=state_dict)
        return transformer

    def load_checkpoint(self, dirname):
        logger.info(f"loading transformer checkpoint from {dirname}")
        dirname = Path(dirname)
        state_dict = torch.load(
            str(dirname / "transformer.pt"), map_location=torch.device("cpu")
        )  # for memory leak on continue
        self.load_state_dict(state_dict=state_dict)
        logger.info(f"loaded transformer checkpoint from {dirname}")
