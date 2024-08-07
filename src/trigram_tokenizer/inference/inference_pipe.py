from typing import Union, Optional, List, Dict, NamedTuple
from pathlib import Path

import torch

from trigram_tokenizer.tokenizer import (
    TrigramTokenizer,
    EncodingBatchInference,
    EncodingInference,
)
from trigram_tokenizer.transformer import TransformerLMHeadModel
from trigram_tokenizer.logging import logger
from trigram_tokenizer.tokenizer.wordsplit import words_to_text


class GenerationResult(NamedTuple):
    uuid: str
    completion: str
    token_ids: List[int]
    tokens: List[str]
    tokens_count_training: List[Optional[int]]
    logits: List[Optional[torch.Tensor]]
    log_probs: Optional[Optional[List[Dict[str, float]]]]
    prompt_token_count_training: List[Optional[int]]


class InferencePipe:
    def __init__(
        self,
        checkpoint_dir: Union[Path, str],
        device: Optional[Union[str, torch.device]] = None,
        reduce_tokenizer_words_to: Optional[int] = None,
        words_target_dir: Optional[Union[str, Path]] = None,
        top_word_dict: Optional[Union[str, Path]] = None,
        overwrite_values: dict = {},
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        logger.info(f"InferencePipe loading from {self.checkpoint_dir}")
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        logger.info(f"InferencePipe setting device {self.device}")
        logger.info(
            f"InferencePipe loading Transformer {self.checkpoint_dir / 'transformer'}"
        )
        overwrite_values.update({"use_flash": False})
        self.transformer: TransformerLMHeadModel = (
            TransformerLMHeadModel.load(
                self.checkpoint_dir / "transformer",
                device=self.device,
                overwrite_values=overwrite_values,
            )
            .eval()
            .cuda()
        )
        logger.info(
            f"InferencePipe loading TrigramTokenizer {self.checkpoint_dir / 'tokenizer'}"
        )
        self.tokenizer: TrigramTokenizer = TrigramTokenizer.load(
            self.checkpoint_dir / "tokenizer",
            reduce_tokenizer_words_to=reduce_tokenizer_words_to,
            words_target_dir=words_target_dir,
            top_word_dict=top_word_dict,
        )
        logger.info(f"InferencePipe loaded")

        assert self.transformer.config.vocab_size == self.tokenizer.config.vocab_size
        assert (
            self.transformer.config.sequence_length
            == self.tokenizer.config.sequence_length
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        encoding: Optional[EncodingInference] = None,
        max_tokens: int = 16,
        echo: bool = False,
        log_probs: Optional[int] = None,
        uuid: str = "",
        abort_eot: bool = True,
        stop_word: Optional[str] = None,
        more_words: Optional[str] = None,
    ):
        token_id_result = list()
        tokens_result = list()
        tokens_count_training: List[Optional[int]] = list()
        prompt_token_count_training: List[Optional[int]] = list()
        logits_result: List[Optional[torch.Tensor]] = list()
        log_probs_result: Optional[List[Dict[str, float]]] = None
        if log_probs is not None:
            log_probs_result = list()

        # prefill stage
        if encoding is not None:
            assert prompt == ""
        else:
            encoding = self.tokenizer.encode(text=prompt)
        batch = EncodingBatchInference.from_encodings([encoding])
        seq_len_decode = batch.position_ids.shape[1]
        logits, kv_cache = self.transformer(
            trigram_set_position_ids=batch.trigram_set_position_ids.detach().to(
                self.device
            ),
            trigram_token_ids=batch.trigram_token_ids.detach().to(self.device),
            trigram_token_ids_offsets=batch.trigram_token_ids_offsets.detach().to(
                self.device
            ),
            position_ids=batch.position_ids.detach().to(self.device).to(torch.long),
            attention_mask=batch.attention_mask.detach().to(self.device),
            kv_cache=dict(),
        )
        if self.tokenizer.config.do_classic_tokenization:
            prompt_token_count_training = []
        else:
            prompt_token_count_training = [
                self.tokenizer.word_counter_full.get(w, 0) for w in encoding.words
            ]

        if echo:
            if self.tokenizer.config.do_classic_tokenization:
                token_id_result = self.tokenizer.classic_tokenizer.encode(prompt).ids
                tokens_result = [
                    self.tokenizer.classic_tokenizer.decode([token_id])
                    for token_id in token_id_result
                ]
                tokens_count_training = [None for _ in tokens_result]

            else:
                token_id_result = [-1 for w in encoding.words]
                tokens_result = [f"{w}" for w in encoding.words]
                tokens_count_training = [
                    self.tokenizer.word_counter_full.get(w, 0) for w in tokens_result
                ]

            logits_result.append(None)  # first token does not get logits
            if log_probs is not None:
                assert log_probs_result is not None
                log_probs_result.append({})  # first token does not get log probs

            decode_result = self.tokenizer.decode(
                logits=logits[0, : len(encoding.words) - 1, :],
                log_probs=log_probs,
                target_words=encoding.words[1:],
                more_words=more_words or prompt,
            )

            logits_result.extend(decode_result.logits)
            if log_probs is not None:
                assert decode_result.log_probs is not None
                assert log_probs_result is not None
                log_probs_result.extend(decode_result.log_probs)

        if max_tokens > 0:
            decode_result = self.tokenizer.decode(
                logits=logits[0, -1:, :],
                log_probs=log_probs,
                more_words=more_words or prompt,
            )
            assert len(decode_result.word_indices) == 1
            token_id_result.append(decode_result.word_indices[0])
            assert len(decode_result.words) == 1
            tokens_result.append(decode_result.words[0])
            tokens_count_training.append(decode_result.words_count_training[0])

            if log_probs is not None:
                assert decode_result.log_probs is not None
                assert len(decode_result.log_probs) == 1
                assert log_probs_result is not None
                log_probs_result.append(decode_result.log_probs[0])

            assert len(decode_result.logits) == 1
            logits_result.append(decode_result.logits[0])

            # decode
            if not (decode_result.words[0] == self.tokenizer.config.end_of_text):
                for _ in range(max_tokens - 1):
                    seq_len_decode += 1

                    if self.tokenizer.config.do_classic_tokenization:
                        dummybatch = EncodingBatchInference.from_encodings(
                            [encoding],
                            pad_attention_mask_to=seq_len_decode,
                        )
                        logits, kv_cache = self.transformer(
                            trigram_set_position_ids=torch.tensor([0]).to(self.device),
                            trigram_token_ids=torch.tensor(
                                [[decode_result.word_indices[0]]]
                            ).to(self.device),
                            trigram_token_ids_offsets=torch.tensor([[0, 1]]).to(
                                self.device
                            ),
                            position_ids=torch.tensor(
                                [[seq_len_decode - 1]],
                                device=self.device,
                                dtype=torch.long,
                            ),
                            attention_mask=dummybatch.attention_mask[
                                :,
                                :,
                                dummybatch.attention_mask.shape[2]
                                - 1 : dummybatch.attention_mask.shape[2],
                                :,
                            ]
                            .detach()
                            .to(self.device),
                            kv_cache=kv_cache,
                        )
                        decode_result = self.tokenizer.decode(
                            logits=logits[0, -1:, :],
                            log_probs=log_probs,
                            more_words=more_words or prompt,
                        )
                    else:
                        next_word = decode_result.words[0]

                        if next_word == "":
                            logger.info(" >> abort - got a nile word")
                            break

                        encoding = self.tokenizer.encode(text=next_word)
                        batch = EncodingBatchInference.from_encodings(
                            [encoding],
                            pad_attention_mask_to=seq_len_decode,
                        )
                        logits, kv_cache = self.transformer(
                            trigram_set_position_ids=batch.trigram_set_position_ids.detach().to(
                                self.device
                            ),
                            trigram_token_ids=batch.trigram_token_ids.detach().to(
                                self.device
                            ),
                            trigram_token_ids_offsets=batch.trigram_token_ids_offsets.detach().to(
                                self.device
                            ),
                            position_ids=torch.tensor(
                                [[seq_len_decode - 1]],
                                device=self.device,
                                dtype=torch.long,
                            ),
                            attention_mask=batch.attention_mask[
                                :,
                                :,
                                batch.attention_mask.shape[2]
                                - 1 : batch.attention_mask.shape[2],
                                :,
                            ]
                            .detach()
                            .to(self.device),
                            kv_cache=kv_cache,
                        )
                        decode_result = self.tokenizer.decode(
                            logits=logits[0, -1:, :],
                            log_probs=log_probs,
                            more_words=more_words or prompt,
                        )

                    assert len(decode_result.word_indices) == 1
                    token_id_result.append(decode_result.word_indices[0])
                    assert len(decode_result.words) == 1
                    tokens_result.append(decode_result.words[0])
                    tokens_count_training.append(decode_result.words_count_training[0])

                    if log_probs is not None:
                        assert decode_result.log_probs is not None
                        assert len(decode_result.log_probs) == 1
                        assert log_probs_result is not None
                        log_probs_result.append(decode_result.log_probs[0])

                    assert len(decode_result.logits) == 1
                    logits_result.append(decode_result.logits[0])

                    if abort_eot and (
                        decode_result.words[0] == self.tokenizer.config.end_of_text
                    ):
                        break

                    if stop_word is not None and (decode_result.words[0] == stop_word):
                        break

        if self.tokenizer.config.do_classic_tokenization:
            completion_result = self.tokenizer.classic_tokenizer.decode(token_id_result)
        else:
            completion_result = words_to_text(tokens_result)

        assert len(token_id_result) == len(tokens_result)
        assert len(token_id_result) == len(logits_result)
        if log_probs:
            assert log_probs_result is not None
            assert len(token_id_result) == len(log_probs_result)

        return GenerationResult(
            uuid=uuid,
            completion=completion_result,
            token_ids=token_id_result,
            tokens=tokens_result,
            tokens_count_training=tokens_count_training,
            logits=logits_result,
            log_probs=log_probs_result,
            prompt_token_count_training=prompt_token_count_training,
        )
