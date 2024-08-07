from typing import Set, List, Optional, Union
from typing_extensions import Never
from dataclasses import dataclass

import torch

from trigram_tokenizer.logging import logger


@dataclass
class EncodingTraining:
    trigram_set_position_ids: torch.Tensor
    trigram_token_ids: torch.Tensor
    trigram_sets_input: List[Set]
    position_ids: torch.Tensor
    trigram_set_input_is_padding: torch.Tensor
    loss_weights: torch.Tensor

    targets: torch.Tensor
    trigram_set_target_is_eot: torch.Tensor
    trigram_sets_targets: List[Set]
    words_targets: List[Optional[str]]

    @staticmethod
    def concat_encodings(
        encodings: List,
        seq_len: Optional[int] = None,
        eot: Optional[int] = None,
        reset_position_ids: bool = True,
    ):
        assert len(encodings) > 0, "cannot concat empty list of encodings"

        trigram_set_position_ids: List[torch.Tensor] = list()
        trigram_token_ids: List[torch.Tensor] = list()
        targets: List[torch.Tensor] = list()
        words_targets: List[Optional[str]] = list()
        position_ids: List[torch.Tensor] = list()
        loss_weights: List[torch.Tensor] = list()
        trigram_set_target_is_eot: List[torch.Tensor] = list()

        trigram_sets_input: List[Union[Set[str], Set[Never]]] = list()
        trigram_sets_targets: List[Union[Set[str], Set[Never]]] = list()

        position_offset = 0
        for encoding in encodings:
            assert isinstance(encoding, EncodingTraining)
            trigram_set_position_ids.append(
                encoding.trigram_set_position_ids + position_offset
            )
            trigram_token_ids.append(encoding.trigram_token_ids)

            assert encoding.targets is not None, f"can only concat training data"
            targets.append(encoding.targets)

            assert encoding.words_targets is not None, f"can only concat training data"
            words_targets.extend(encoding.words_targets)

            loss_weights.append(encoding.loss_weights)
            if reset_position_ids:
                position_ids.append(encoding.position_ids)
            else:
                position_ids.append(encoding.position_ids + position_offset)

            assert (
                encoding.trigram_set_target_is_eot is not None
            ), f"can only concat training data"
            trigram_set_target_is_eot.append(encoding.trigram_set_target_is_eot)

            trigram_sets_input.extend(encoding.trigram_sets_input)

            assert (
                encoding.trigram_sets_targets is not None
            ), f"can only concat training data"
            trigram_sets_targets.extend(encoding.trigram_sets_targets)

            position_offset += len(encoding.position_ids)

        # cat
        trigram_set_position_ids_tensor = torch.cat(trigram_set_position_ids, dim=0)
        trigram_token_ids_tensor = torch.cat(trigram_token_ids, dim=0)
        targets_tensor = torch.cat(targets, dim=0)
        loss_weights_tensor = torch.cat(loss_weights, dim=0)
        position_ids_tensor = torch.cat(position_ids, dim=0)
        trigram_set_target_is_eot_tensor = torch.cat(trigram_set_target_is_eot, dim=0)
        trigram_set_input_is_padding = torch.zeros_like(
            trigram_set_target_is_eot_tensor
        )

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
                targets_tensor = targets_tensor[:seq_len, :]
                words_targets = words_targets[:seq_len]
                position_ids_tensor = position_ids_tensor[:seq_len]
                loss_weights_tensor = loss_weights_tensor[:seq_len]
                trigram_set_target_is_eot_tensor = trigram_set_target_is_eot_tensor[
                    :seq_len
                ]
                trigram_sets_input = trigram_sets_input[:seq_len]
                trigram_sets_targets = trigram_sets_targets[:seq_len]
                trigram_set_input_is_padding = torch.zeros_like(
                    trigram_set_target_is_eot_tensor
                )
            elif len(position_ids_tensor) < seq_len:
                # pad
                padding_size = seq_len - len(position_ids_tensor)
                # logger.info(f"EncodingTraining PADDING: {padding_size}")

                trigram_set_input_is_padding = torch.cat(
                    [
                        torch.zeros((len(position_ids_tensor),), dtype=torch.bool),
                        torch.ones((padding_size,), dtype=torch.bool),
                    ],
                    dim=0,
                )[:seq_len]

                # these remain unchanged as they will only fill positions later
                # trigram_set_position_ids_tensor
                # trigram_token_ids_tensor

                if eot:
                    pad_eot = torch.ones(
                        (padding_size, targets_tensor.shape[1]),
                        dtype=targets_tensor.dtype,
                        device=targets_tensor.device,
                    )
                    pad_eot *= eot
                else:
                    pad_eot = torch.zeros(
                        (padding_size, targets_tensor.shape[1]),
                        dtype=targets_tensor.dtype,
                        device=targets_tensor.device,
                    )

                targets_tensor = torch.cat(
                    [
                        targets_tensor,
                        pad_eot,
                    ],
                    dim=0,
                )[:seq_len, :]
                position_ids_tensor = torch.arange(0, seq_len)
                loss_weights_tensor = torch.cat(
                    [
                        loss_weights_tensor,
                        torch.zeros(
                            (padding_size,),
                            dtype=loss_weights_tensor.dtype,
                            device=loss_weights_tensor.device,
                        ),
                    ],
                    dim=0,
                )[:seq_len]
                trigram_set_target_is_eot_tensor = torch.cat(
                    [
                        trigram_set_target_is_eot_tensor,
                        torch.ones(
                            (padding_size,),
                            dtype=trigram_set_target_is_eot_tensor.dtype,
                            device=trigram_set_target_is_eot_tensor.device,
                        ),
                    ],
                    dim=0,
                )[:seq_len]
                words_targets = words_targets + [None for _ in range(padding_size)]
                words_targets = words_targets[:seq_len]
                trigram_sets_input = (
                    trigram_sets_input + [set() for _ in range(padding_size)][:seq_len]
                )
                trigram_sets_targets = (
                    trigram_sets_targets
                    + [set() for _ in range(padding_size)][:seq_len]
                )

        return EncodingTraining(
            trigram_set_position_ids=trigram_set_position_ids_tensor,
            trigram_token_ids=trigram_token_ids_tensor,
            targets=targets_tensor,
            trigram_sets_input=trigram_sets_input,
            trigram_sets_targets=trigram_sets_targets,
            words_targets=words_targets,
            position_ids=position_ids_tensor,
            loss_weights=loss_weights_tensor,
            trigram_set_target_is_eot=trigram_set_target_is_eot_tensor,
            trigram_set_input_is_padding=trigram_set_input_is_padding,
        )

    def __eq__(self, o):
        assert isinstance(o, EncodingTraining)
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

        if (
            not self.trigram_set_input_is_padding.shape
            == o.trigram_set_input_is_padding.shape
        ):
            return False
        if (
            not (self.trigram_set_input_is_padding == o.trigram_set_input_is_padding)
            .all()
            .item()
        ):
            return False

        if not self.loss_weights.shape == o.loss_weights.shape:
            return False
        if not (self.loss_weights == o.loss_weights).all().item():
            return False

        if not (self.targets is None and o.targets is None):
            if self.targets is None or o.targets is None:
                return False
            if not self.targets.shape == o.targets.shape:
                return False
            if not (self.targets == o.targets).all().item():
                return False

        if not (
            self.trigram_set_target_is_eot is None
            and o.trigram_set_target_is_eot is None
        ):
            if (
                self.trigram_set_target_is_eot is None
                or o.trigram_set_target_is_eot is None
            ):
                return False
            if (
                not self.trigram_set_target_is_eot.shape
                == o.trigram_set_target_is_eot.shape
            ):
                return False
            if (
                not (self.trigram_set_target_is_eot == o.trigram_set_target_is_eot)
                .all()
                .item()
            ):
                return False

        if not self.trigram_sets_targets == o.trigram_sets_targets:
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

        if self.targets is None:
            targets_tensor = None
        else:
            targets_tensor = self.targets[idx_slice.start : idx_slice.stop, :]

        if self.words_targets is None:
            words_targets = None
        else:
            words_targets = self.words_targets[idx_slice.start : idx_slice.stop]

        position_ids_tensor = (
            self.position_ids[idx_slice.start : idx_slice.stop]
            - self.position_ids[idx_slice.start]
        )
        loss_weights_tensor = self.loss_weights[idx_slice.start : idx_slice.stop]
        if self.trigram_set_target_is_eot is None:
            trigram_set_target_is_eot_tensor = None
        else:
            trigram_set_target_is_eot_tensor = self.trigram_set_target_is_eot[
                idx_slice.start : idx_slice.stop
            ]
        trigram_set_input_is_padding = self.trigram_set_input_is_padding[
            idx_slice.start : idx_slice.stop
        ]
        trigram_sets_input = self.trigram_sets_input[idx_slice.start : idx_slice.stop]
        if self.trigram_sets_targets is None:
            trigram_sets_targets = None
        else:
            trigram_sets_targets = self.trigram_sets_targets[
                idx_slice.start : idx_slice.stop
            ]

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

        return EncodingTraining(
            trigram_set_position_ids=trigram_set_position_ids_tensor,
            trigram_token_ids=trigram_token_ids_tensor,
            targets=targets_tensor,
            words_targets=words_targets,
            trigram_sets_input=trigram_sets_input,
            trigram_sets_targets=trigram_sets_targets,
            position_ids=position_ids_tensor,
            loss_weights=loss_weights_tensor,
            trigram_set_target_is_eot=trigram_set_target_is_eot_tensor,
            trigram_set_input_is_padding=trigram_set_input_is_padding,
        )
