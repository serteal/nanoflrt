import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nanoflrt

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fill in your own message and target
messages = [{"role": "user", "content": "Give me instructions for how to ..."}]
target = "Sure, here are some instructions for how to ..."

config = nanoflrt.FLRTConfig(
    use_kv_cache=False,
)
result = nanoflrt.run(model, tokenizer, messages, target, config)

# Generate with the best FLRT string
messages[-1]["content"] = messages[-1]["content"] + result.best_string

input = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
output = model.generate(input, do_sample=False, max_new_tokens=512)

print(f"Prompt:\n{messages[-1]['content']}\n")
print(
    f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1] :], skip_special_tokens=True)[0]}"
)
