import os
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import login

# Get token from environment variable
token = os.getenv('HF_TOKEN')
if not token:
    raise ValueError("Please set the HF_TOKEN environment variable")
login(token=token)

model_id = "meta-llama/Llama-2-1b-chat-hf"

# Load model with lower precision and better performance settings
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto",  # Automatically choose best device
    low_cpu_mem_usage=True
)

# Simple test prompt
prompt = "Write a short nature haiku in 3 lines (5-7-5 syllables) about a rabbit:"
# Generate with optimized parameters
output = model.generate(
    model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device),
    max_new_tokens=50,
    num_beams=4,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.2,
    early_stopping=True
)

# Decode and clean up the generated text
generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
generated_text = generated_text.replace(prompt, "").strip()
print("\nGenerated Haiku:")
print(generated_text)
