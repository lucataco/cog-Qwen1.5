# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import torch
import subprocess
from threading import Thread
from transformers.generation import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8"
MODEL_CACHE = "checkpoints"

MODEL_URL_MAP = {
    "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8": "https://weights.replicate.delivery/default/qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8.tar",
    "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8": "https://weights.replicate.delivery/default/qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8.tar",
    "Qwen/Qwen1.5-4B-Chat-GPTQ-Int8": "https://weights.replicate.delivery/default/qwen/Qwen1.5-4B-Chat-GPTQ-Int8.tar",
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": "https://weights.replicate.delivery/default/qwen/Qwen1.5-7B-Chat-GPTQ-Int8.tar",
    "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8": "https://weights.replicate.delivery/default/qwen/Qwen1.5-14B-Chat-GPTQ-Int8.tar",
    "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8": "https://weights.replicate.delivery/default/qwen/Qwen1.5-72B-Chat-GPTQ-Int8.tar",
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights if they are not already cached
        if not os.path.exists(MODEL_CACHE):
            checkpoint_url = MODEL_URL_MAP[MODEL_NAME]
            download_weights(checkpoint_url, MODEL_CACHE)
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE
        ).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="Give me a short introduction to large language model."),
        system_prompt: str = Input(description="System prompt", default="You are a helpful assistant."),
        max_new_tokens: int = Input(description="The maximum number of tokens to generate", default=512, ge=1, le=32768),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            default=1.0,ge=0.1,le=5.0,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.",
            default=1.0,ge=0.01,le=1.0,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens.",
            default=1,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            default=1.0,ge=0.01,le=10.0,
        ),
        seed: int = Input(description="The seed for the random number generator", default=None),
    ) -> ConcatenateIterator:
        """Run a single prediction on the model"""
        if seed == None:
            seed = torch.randint(0, 2**30, (1,)).item()
        torch.random.manual_seed(seed)
        print("Using seed:", seed)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        tokens = self.tokenizer([text], return_tensors="pt")
        input_ids = tokens.input_ids.to('cuda')
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "streamer": streamer
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for _, new_text in enumerate(streamer):
            yield new_text
        thread.join()
