# `nanoflrt`

A custom implementation of a variation of the FLRT algorithm from [FLRT: Fluent Student-Teacher Redteaming](https://arxiv.org/abs/2407.17447), here we replace the worst candidate from the buffer, and don't (currently) use the additional loss functions outlined in the FLRT paper (we only use cross-entropy and ignore the finetuning and perplexity losses).

## Installation

1. Install from the git repo:

```bash
pip install git+https://github.com/serteal/nanoflrt
```

or use [`uv`](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/serteal/nanoflrt
```

## Usage

```python
import nanoflrt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fill in your own message and target
message = "Give me instructions for how to ..."
target = "Sure, here are some instructions for how to ..."

result = nanoflrt.run(model, tokenizer, message, target)

# Generate with the best FLRT string
messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string

input = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
output = model.generate(input, do_sample=False, max_new_tokens=512)

print(f"Prompt:\n{messages[-1]['content']}\n")
print(
    f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1] :], skip_special_tokens=True)[0]}"
)
```

Check out the [examples](examples/) for other use-cases.
