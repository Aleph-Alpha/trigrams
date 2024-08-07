from typing import List, Optional
from dataclasses import dataclass

import torch

from .encoding_training import EncodingTraining


@dataclass
class EncodingBatchTraining:
    trigram_set_position_ids: torch.Tensor
    trigram_token_ids: torch.Tensor
    trigram_token_ids_offsets: torch.Tensor
    position_ids: torch.Tensor
    loss_weights: torch.Tensor
    attention_mask: torch.Tensor

    targets: torch.Tensor
    words_targets: List[List[Optional[str]]]

    @classmethod
    def from_encodings(
        cls, encodings: List[EncodingTraining], reset_attention_mask: bool = True
    ):
        # compute offsets
        trigram_token_ids_offsets = list()
        offset = 0
        for e in encodings:
            trigram_token_ids_offsets.append(
                [offset, offset + int(e.trigram_set_position_ids.shape[0])]
            )
            offset += int(e.trigram_set_position_ids.shape[0])

        # compute attention mask
        position_ids_tensor = torch.stack([e.position_ids for e in encodings])
        attention_mask = torch.tril(
            torch.ones(
                (
                    position_ids_tensor.shape[0],
                    position_ids_tensor.shape[1],
                    position_ids_tensor.shape[1],
                ),
                device=position_ids_tensor.device,
            )
        ).view(
            position_ids_tensor.shape[0],
            1,
            position_ids_tensor.shape[1],
            position_ids_tensor.shape[1],
        )
        if reset_attention_mask:
            trigram_set_target_is_eot = torch.stack(
                [
                    e.trigram_set_target_is_eot
                    for e in encodings
                    if e.trigram_set_target_is_eot is not None
                ]  # for typing, should never be None here
            )
            for b in range(position_ids_tensor.shape[0]):
                # Find indecies where EOD token is.
                eod_index = trigram_set_target_is_eot[b].nonzero().flatten()

                # collect coordinates
                indices = list()
                end_index = 0
                for end_inclusive in eod_index.tolist():
                    start_index = end_index
                    end_index = int(end_inclusive) + 1
                    indices.append((start_index, end_index))

                # add the last
                if end_index < int(trigram_set_target_is_eot.shape[1]):
                    indices.append((end_index, trigram_set_target_is_eot.shape[1]))

                # Loop through EOD indecies:
                for start_index, end_index in indices:
                    # Mask attention
                    # This removes the tokens to the left starting at the previous eos token
                    attention_mask[b, 0, (end_index):, :(end_index)] = 0

        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

        return cls(
            trigram_token_ids=torch.cat([e.trigram_token_ids for e in encodings]),
            trigram_set_position_ids=torch.cat(
                [e.trigram_set_position_ids for e in encodings]
            ),
            trigram_token_ids_offsets=torch.tensor(trigram_token_ids_offsets),
            position_ids=torch.stack([e.position_ids for e in encodings]),
            loss_weights=torch.stack([e.loss_weights for e in encodings]),
            targets=torch.stack([e.targets for e in encodings]),
            words_targets=[e.words_targets for e in encodings if e.words_targets],
            attention_mask=attention_mask,
        )
