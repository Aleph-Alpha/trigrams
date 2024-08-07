from typing import Set, List, Optional
from dataclasses import dataclass

import torch

from .encoding_inference import EncodingInference


@dataclass
class EncodingBatchInference:
    trigram_set_position_ids: torch.Tensor
    trigram_token_ids: torch.Tensor
    trigram_token_ids_offsets: torch.Tensor
    trigram_sets_input: List[List[Set]]
    position_ids: torch.Tensor
    attention_mask: torch.Tensor

    words: List[List[str]]

    @classmethod
    def from_encodings(
        cls,
        encodings: List[EncodingInference],
        pad_attention_mask_to: Optional[int] = None,
    ):
        # compute offsets
        trigram_token_ids_offsets = list()
        words = list()
        offset = 0
        for e in encodings:
            trigram_token_ids_offsets.append(
                [offset, offset + int(e.trigram_set_position_ids.shape[0])]
            )
            offset += int(e.trigram_set_position_ids.shape[0])
            words.append(e.words)

        # compute attention mask
        position_ids_tensor = torch.stack([e.position_ids for e in encodings])
        attention_mask = torch.tril(
            torch.ones(
                (
                    position_ids_tensor.shape[0],
                    (
                        position_ids_tensor.shape[1]
                        if pad_attention_mask_to is None
                        else pad_attention_mask_to
                    ),
                    (
                        position_ids_tensor.shape[1]
                        if pad_attention_mask_to is None
                        else pad_attention_mask_to
                    ),
                ),
                device=position_ids_tensor.device,
            )
        ).view(
            position_ids_tensor.shape[0],
            1,
            (
                position_ids_tensor.shape[1]
                if pad_attention_mask_to is None
                else pad_attention_mask_to
            ),
            (
                position_ids_tensor.shape[1]
                if pad_attention_mask_to is None
                else pad_attention_mask_to
            ),
        )

        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

        return cls(
            trigram_token_ids=torch.cat([e.trigram_token_ids for e in encodings]),
            trigram_set_position_ids=torch.cat(
                [e.trigram_set_position_ids for e in encodings]
            ),
            trigram_token_ids_offsets=torch.tensor(trigram_token_ids_offsets),
            trigram_sets_input=[e.trigram_sets_input for e in encodings],
            position_ids=torch.stack([e.position_ids for e in encodings]),
            attention_mask=attention_mask,
            words=words,
        )
