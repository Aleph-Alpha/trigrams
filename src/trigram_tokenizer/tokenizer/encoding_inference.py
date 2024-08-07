from typing import Set, List, Optional, Union

from dataclasses import dataclass

import torch


@dataclass
class EncodingInference:
    trigram_set_position_ids: torch.Tensor
    trigram_token_ids: torch.Tensor
    trigram_sets_input: List[Set]
    words: List[str]
    position_ids: torch.Tensor

    @staticmethod
    def concat_encodings(encodings: List, seq_len: Optional[int] = None):
        assert len(encodings) > 0, "cannot concat empty list of encodings"

        trigram_set_position_ids: List[torch.Tensor] = list()
        trigram_token_ids: List[torch.Tensor] = list()
        position_ids: List[torch.Tensor] = list()

        trigram_sets_input: List[Set[str]] = list()
        words = list()

        position_offset = 0
        for encoding in encodings:
            assert isinstance(encoding, EncodingInference)
            trigram_set_position_ids.append(
                encoding.trigram_set_position_ids + position_offset
            )
            trigram_token_ids.append(encoding.trigram_token_ids)

            position_ids.append(encoding.position_ids + position_offset)

            trigram_sets_input.extend(encoding.trigram_sets_input)

            position_offset += len(encoding.position_ids)

            words.extend(encoding.words)

        # cat
        trigram_set_position_ids_tensor = torch.cat(trigram_set_position_ids, dim=0)
        trigram_token_ids_tensor = torch.cat(trigram_token_ids, dim=0)
        position_ids_tensor = torch.cat(position_ids, dim=0)

        # pad or cut
        if seq_len is not None:
            if len(position_ids_tensor) > seq_len:
                # cut
                trigram_set_position_ids_tensor = trigram_set_position_ids_tensor[
                    trigram_set_position_ids_tensor < seq_len
                ]
                trigram_token_ids_tensor = trigram_token_ids_tensor[
                    : len(trigram_set_position_ids_tensor), :
                ]
                position_ids_tensor = position_ids_tensor[:seq_len]
                trigram_sets_input = trigram_sets_input[:seq_len]
                words = words[:seq_len]
            elif len(position_ids_tensor) < seq_len:
                # pad
                padding_size = seq_len - len(position_ids_tensor)
                position_ids_tensor = torch.arange(0, seq_len)
                trigram_sets_input = trigram_sets_input + [
                    set() for _ in range(padding_size)
                ]
                words = words + ["" for _ in range(padding_size)]

        return EncodingInference(
            trigram_set_position_ids=trigram_set_position_ids_tensor,
            trigram_token_ids=trigram_token_ids_tensor,
            trigram_sets_input=trigram_sets_input,
            position_ids=position_ids_tensor,
            words=words,
        )

    def __eq__(self, o):
        assert isinstance(o, EncodingInference)
        if not self.trigram_set_position_ids.shape == o.trigram_set_position_ids.shape:
            return False
        if (
            not (self.trigram_set_position_ids == o.trigram_set_position_ids)
            .all()
            .item()
        ):
            return False

        if not self.trigram_token_ids.shape == o.trigram_token_ids.shape:
            return False
        if not (self.trigram_token_ids == o.trigram_token_ids).all().item():
            return False

        if not self.trigram_sets_input == o.trigram_sets_input:
            return False

        if not self.position_ids.shape == o.position_ids.shape:
            return False
        if not (self.position_ids == o.position_ids).all().item():
            return False

        return True

    def __len__(self):
        return len(self.position_ids)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            idx_slice = slice(idx.start or 0, idx.stop or len(self))
        elif isinstance(idx, int):
            idx_slice = slice(idx, idx + 1)
        else:
            raise NotImplementedError

        if idx_slice.stop > len(self):
            idx_slice = slice(idx_slice.start, len(self))
        if idx_slice.start < 0:
            raise IndexError

        trigram_sets_input = self.trigram_sets_input[idx_slice.start : idx_slice.stop]
        words = self.words[idx_slice.start : idx_slice.stop]

        position_ids_tensor = (
            self.position_ids[idx_slice.start : idx_slice.stop]
            - self.position_ids[idx_slice.start]
        )

        trigram_set_position_ids_tensor = (
            self.trigram_set_position_ids[
                (self.trigram_set_position_ids >= idx_slice.start).logical_and(
                    self.trigram_set_position_ids < idx_slice.stop
                )
            ]
            - self.position_ids[idx_slice.start]
        )
        trigram_token_ids_tensor = self.trigram_token_ids[
            (self.trigram_set_position_ids >= idx_slice.start).logical_and(
                self.trigram_set_position_ids < idx_slice.stop
            )
        ]

        return EncodingInference(
            trigram_set_position_ids=trigram_set_position_ids_tensor,
            trigram_token_ids=trigram_token_ids_tensor,
            trigram_sets_input=trigram_sets_input,
            position_ids=position_ids_tensor,
            words=words,
        )
