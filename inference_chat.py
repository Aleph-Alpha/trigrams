import json
from tqdm import tqdm

from trigram_tokenizer.inference import InferencePipe
from trigram_tokenizer.logging import logger


def generate(
    prompt: str,
    max_tokens: int = 4,
    echo: bool = False,
):
    result = inference_pipe.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        echo=echo,
        log_probs=10_000,
    )

    logger.info("#" * 50)
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"COMPLETION: {result.completion}")
    logger.info(f"COMPLETION: {result.tokens}")
    logger.info(f"COMPLETION: {result.tokens_count_training}")
    logger.info("#" * 50)
    return result


if __name__ == "__main__":
    inference_pipe = InferencePipe(
        "<some checkpoint>",  
        reduce_tokenizer_words_to=600000,
    )

    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYour task as a language model is to help users. Provide responses that are engaging, clear, and structured.<|endoftext|>"
    prompt_for_reset = prompt
    print("#" * 10)
    print("# Type 'STOP' to stop and 'RESET' to reset")
    print("#" * 10)
    while True:
        text = input("User: ")
        if text == "STOP":
            break
        if text == "RESET":
            prompt = prompt_for_reset
            continue

        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|endoftext|><|start_header_id|>assistant<|end_header_id|>"

        response = inference_pipe.generate(
            prompt=prompt,
            max_tokens=512,
            echo=False,
        )
        print("Trigram-7b-instruct:", response.completion.replace("<|endoftext|>", ""))
        
        prompt += response.completion
