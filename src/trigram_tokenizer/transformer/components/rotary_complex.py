from typing import Tuple, Optional, Union
import torch


from einops import rearrange, repeat


def vector_gather(vectors: torch.Tensor, indices: torch.Tensor):
    """
    Gathers (batched) vectors according to indices.
    """
    vectors = repeat(vectors, "sq d -> sq B nh d", B=indices.shape[1], nh=1)
    indices = repeat(
        indices,
        "sq b -> sq b nh d",
        nh=1,
        d=vectors.shape[-1],
    )

    out = torch.gather(vectors, dim=0, index=indices)

    out = rearrange(out, "sq b nh hh -> b sq nh hh")

    return out


def precompute_freqs_cis(
    dim: int, end: int, theta: float, device: Optional[Union[str, torch.device]]
):
    theta = float(theta)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    if device is not None:
        freqs_cis = freqs_cis.to(device)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape[0] == x.shape[1]
    assert freqs_cis.shape[1] == x.shape[-1]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    query_position_ids: Optional[torch.Tensor],
    key_position_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    if query_position_ids is None:
        freqs_cis_q_ = reshape_for_broadcast(freqs_cis, xq_)
    else:
        freqs_cis_q_ = vector_gather(freqs_cis, query_position_ids)

    if key_position_ids is None:
        freqs_cis_k_ = reshape_for_broadcast(freqs_cis, xq_)
    else:
        freqs_cis_k_ = vector_gather(freqs_cis, key_position_ids)

    xq_out = torch.view_as_real(xq_ * freqs_cis_q_).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k_).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbeddingComplex(torch.nn.Module):
    """
    Relative rotary position embedding based on
    * RoFormer: Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
    * Rotary Embeddings: A Relative Revolution (https://blog.eleuther.ai/rotary-embeddings/)
    """

    def __init__(
        self,
        dimensions: int,
        base: int,
        max_seq_length: int,
        device: Optional[Union[str, torch.device]],
    ):
        super().__init__()
        assert (
            dimensions > 1
        ), "RotaryEmbedding cannot use `dim` == 1, this results in weird reshape errors"

        self.freqs_cis = precompute_freqs_cis(
            dim=dimensions, end=max_seq_length, theta=base, device=device
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_position_ids: Optional[torch.Tensor] = None,
        key_position_ids: Optional[torch.Tensor] = None,
    ):
        query, key = apply_rotary_emb(
            xq=rearrange(query, "sq b nh hh -> b sq nh hh"),
            xk=rearrange(key, "sq b nh hh -> b sq nh hh"),
            freqs_cis=self.freqs_cis,
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
        )
        return rearrange(query, "b sq nh hh -> sq b nh hh"), rearrange(
            key, "b sq nh hh -> sq b nh hh"
        )
