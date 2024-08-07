import pytest
from trigram_tokenizer.tokenizer import (
    TrigramTokenizer,
    TrigramTokenizerConfig,
)


@pytest.fixture
def trigram_tokenizer():
    config = TrigramTokenizerConfig.from_dict(
        {
            "lowercase": False,
            "vocab_size": 32000,
            "vocab_population": 4,
            "seed": 42,
            "end_of_text": "<|endoftext|>",
            "cache_dir": "tmp/tokenizer",
            "sequence_length": 2048,
        }
    )
    tokenizer = TrigramTokenizer.init(config=config)
    yield tokenizer
