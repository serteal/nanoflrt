"""
Simple script that runs FLRT with the default settings
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nanoflrt
from nanoflrt import FLRTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    messages = [{"role": "user", "content": args.prompt}]

    config = FLRTConfig()
    result = nanoflrt.run(
        model,
        tokenizer,
        messages,
        args.target,
        config,
    )

    messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string

    input = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(args.device)
    output = model.generate(input, do_sample=False, max_new_tokens=512)

    print(f"Prompt:\n{messages[-1]['content']}\n")
    print(
        f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1] :], skip_special_tokens=True)[0]}"
    )


if __name__ == "__main__":
    main()
