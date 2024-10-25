import os
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
from huggingface_hub import login

# Get token from environment variable
token = os.getenv('HF_TOKEN')
if not token:
    raise ValueError("Please set the HF_TOKEN environment variable")
login(token=token)

# Initialize the pipeline with Llama 3.2
pipe = pipeline("text-generation", 
    model="meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Simple test prompt
prompt = "Write a short nature haiku in 3 lines (5-7-5 syllables) about a rabbit:"
# Generate text using the pipeline
output = pipe(prompt, 
    max_new_tokens=50,
    num_beams=4,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.2
)

# Extract the generated text
generated_text = output[0]['generated_text'].replace(prompt, "").strip()
print("\nGenerated Haiku:")
print(generated_text)
