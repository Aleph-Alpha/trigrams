from typing import Optional, Union, Dict, Tuple
import math
import torch
from einops import rearrange

from ..config import TransformerArchitectureConfig
from .rotary_complex import RotaryEmbeddingComplex

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except:
    pass


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    query_key_scaling_factor: Optional[float],
    dropout_attention_probs: float,
    local_attention_window_size: Optional[int] = None,
) -> torch.Tensor:
    attention_mask = attention_mask.to(torch.int32)
    """
    Compute attention via flash attention dependency.

    Args:
        query (Tensor [s_q b h])
        key (Tensor [s_k b h])
        value (Tensor [s_k b h])
    Returns:
        Tensor[s_q, b, hp] : context_layer
        Tensor[b np s_q s_k] : attention_probs
    # """
    assert (
        flash_attn_varlen_func is not None
    ), "Please install Flash Attention via optimization requirements"
    assert (
        attention_mask.dtype == torch.int32
    ), f"flash attention needs an attention mask with one dimension and dtype int32, got shape {attention_mask.shape} and dtype {attention_mask.dtype}"

    batch_size = query.shape[1]

    # reshape into format expected by flash attention [sq, b, np, hn] => [b * sq, np, hn]
    query = rearrange(query, "s_q b n h -> (b s_q) n h")
    key = rearrange(key, "s_k b n h -> (b s_k) n h")
    value = rearrange(value, "s_k b n h -> (b s_k) n h")

    if local_attention_window_size is None:
        local_attention_window_size = -1

    cumulative_seq_len_q = attention_mask
    cumulative_seq_len_k = attention_mask
    max_seq_len_q = (attention_mask[1:] - attention_mask[:-1]).max().item()
    max_seq_len_k = max_seq_len_q

    # breakpoint()

    attention_output = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cumulative_seq_len_q,
        cu_seqlens_k=cumulative_seq_len_k,
        max_seqlen_q=max_seq_len_q,
        max_seqlen_k=max_seq_len_k,
        dropout_p=dropout_attention_probs,
        softmax_scale=query_key_scaling_factor,
        causal=True,
        window_size=(local_attention_window_size, local_attention_window_size),
    )

    attention_output = rearrange(
        attention_output, "(b s) n h -> b s (n h)", b=batch_size
    )

    return attention_output


class Attention(torch.nn.Module):
    def __init__(
        self,
        config: TransformerArchitectureConfig,
        layer_index: int,
        device: Optional[Union[str, torch.device]],
    ):
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        self.hidden_size_per_attention_head = (
            self.config.hidden_size / self.config.num_attention_heads
        )

        self.scaling_factor: float = 1 / math.sqrt(
            self.config.hidden_size / self.config.num_attention_heads
        )

        self.rotary = RotaryEmbeddingComplex(
            dimensions=self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.rotary_embedding_base,
            max_seq_length=self.config.sequence_length,
            device=device,
        )

        self.query = torch.nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            dtype=self.config.precision.dtype,
            device=device,
            bias=self.config.bias_terms,
        )

        self.key = torch.nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            dtype=self.config.precision.dtype,
            device=device,
            bias=self.config.bias_terms,
        )

        self.value = torch.nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            dtype=self.config.precision.dtype,
            device=device,
            bias=self.config.bias_terms,
        )

        self.dense = torch.nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            dtype=self.config.precision.dtype,
            device=device,
            bias=self.config.bias_terms,
        )

        if self.config.init == "xavier":
            torch.nn.init.xavier_normal_(self.query.weight)
            torch.nn.init.xavier_normal_(self.key.weight)
            torch.nn.init.xavier_normal_(self.value.weight)
            torch.nn.init.xavier_normal_(self.dense.weight)
        elif self.config.init == "pythia":
            std = math.sqrt(2 / (5 * self.config.hidden_size))
            torch.nn.init.normal_(self.query.weight, mean=0.0, std=std)
            torch.nn.init.normal_(self.key.weight, mean=0.0, std=std)
            torch.nn.init.normal_(self.value.weight, mean=0.0, std=std)
            std = 2 / self.config.num_layers / math.sqrt(self.config.hidden_size)
            torch.nn.init.normal_(self.dense.weight, mean=0.0, std=std)
        else:
            assert self.config.init == "normal"
            torch.nn.init.normal_(
                self.query.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )

            torch.nn.init.normal_(
                self.key.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )
            torch.nn.init.normal_(
                self.value.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )

            torch.nn.init.normal_(
                self.dense.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        batch_size = hidden_states.shape[0]
        query = rearrange(
            self.query(hidden_states),
            "b sq (np hn) -> sq b np hn",
            np=self.config.num_attention_heads,
        )
        key = rearrange(
            self.key(hidden_states),
            "b sq (np hn) -> sq b np hn",
            np=self.config.num_attention_heads,
        )
        value = rearrange(
            self.value(hidden_states),
            "b sq (np hn) -> sq b np hn",
            np=self.config.num_attention_heads,
        )

        # breakpoint()

        position_ids = rearrange(position_ids, "b sq -> sq b")
        query, key = self.rotary(
            query=query,
            key=key,
            query_position_ids=position_ids,
            key_position_ids=position_ids,
        )

        if self.config.use_flash:
            pass  # nothing to be done
        else:
            query = rearrange(query, "s_q b n h -> s_q (b n) h")
            key = rearrange(key, "s_k b n h -> s_k (b n) h")
            value = rearrange(value, "s_k b n h -> s_k (b n) h")

        if kv_cache is not None:
            current_cache = kv_cache.get(self.layer_index)
            if current_cache is not None:
                past_key, past_value = current_cache
                key = torch.cat((past_key, key), dim=0)
                value = torch.cat((past_value, value), dim=0)

            kv_cache[self.layer_index] = (key, value)

        if self.config.use_flash:
            hidden_states = flash_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                query_key_scaling_factor=self.scaling_factor,
                dropout_attention_probs=0.0,
            )

        else:
            matmul_result = (
                torch.matmul(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2))
                * self.scaling_factor
            )
            attention_scores = rearrange(
                matmul_result, "(b n) s_q s_k -> b n s_q s_k", b=batch_size
            )

            # softmax
            attention_scores_dtype = attention_scores.dtype
            attention_scores = attention_scores.float()
            attention_scores.masked_fill_(
                attention_mask.to(attention_scores.device), -10000.0
            )
            attention_probs = torch.nn.functional.softmax(attention_scores, -1)
            attention_probs = attention_probs.to(attention_scores_dtype)
            attention_probs = rearrange(attention_probs, "b n s_q s_k -> (b n) s_q s_k")

            # multiply with values
            hidden_states = torch.bmm(
                attention_probs.to(dtype=value.dtype), value.transpose(0, 1)
            )
            hidden_states = rearrange(
                hidden_states, "(b np) sq hn -> b sq (np hn)", b=batch_size
            )

        hidden_states = self.dense(hidden_states)

        return hidden_states
