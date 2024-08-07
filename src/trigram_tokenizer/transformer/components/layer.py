from typing import Optional, Union, Dict, Tuple
import torch

from ..config import TransformerArchitectureConfig
from .attention import Attention
from .mlp import MLP

from .rmsnorm import RMSNorm


class Layer(torch.nn.Module):
    def __init__(
        self,
        config: TransformerArchitectureConfig,
        layer_index: int,
        device: Optional[Union[str, torch.device]],
    ):
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        self.input_layernorm: Union[torch.nn.LayerNorm, RMSNorm]
        if self.config.norm == "layer":
            self.input_layernorm = torch.nn.LayerNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                dtype=self.config.precision.dtype,
                device=device,
            )
        else:
            assert self.config.norm == "rms"
            self.input_layernorm = RMSNorm(
                dimensions=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                dtype=self.config.precision.dtype,
                device=device,
            )
        self.attention = Attention(
            config=config, layer_index=layer_index, device=device
        )

        self.post_attention_layernorm: Union[torch.nn.LayerNorm, RMSNorm]
        if self.config.norm == "layer":
            self.post_attention_layernorm = torch.nn.LayerNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                dtype=self.config.precision.dtype,
                device=device,
            )
        else:
            assert self.config.norm == "rms"
            self.post_attention_layernorm = RMSNorm(
                dimensions=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                dtype=self.config.precision.dtype,
                device=device,
            )
        self.mlp = MLP(config=config, layer_index=layer_index, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        # attention block
        hidden_states_tmp = self.input_layernorm(hidden_states)
        hidden_state_tmp = self.attention(
            hidden_states_tmp,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + hidden_state_tmp

        # mlp block
        hidden_states_tmp = self.post_attention_layernorm(hidden_states)
        hidden_states_tmp = self.mlp(
            hidden_states_tmp,
        )
        hidden_states = hidden_states + hidden_states_tmp

        return hidden_states
