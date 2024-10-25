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

# Initialize the pipeline with OPT model
pipe = pipeline("text-generation", 
    model="facebook/opt-350m",
    torch_dtype=torch.float16,
    device_map="auto"
)

try:
    # Simple test prompt
    prompt = "Write a short nature haiku in 3 lines (5-7-5 syllables) about a rabbit:"
    print("\nGenerating haiku...")
    
    # Generate text using the pipeline
    output = pipe(prompt, 
        max_new_tokens=30,  # Reduced for haiku
        num_beams=5,
        temperature=0.9,
        do_sample=True,
        top_p=0.95,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )

    # Extract and format the generated text
    generated_text = output[0]['generated_text'].replace(prompt, "").strip()
    print("\nGenerated Haiku:")
    for line in generated_text.split('\n')[:3]:  # Only take first 3 lines
        print(line.strip())

except Exception as e:
    print(f"\nError generating text: {str(e)}")
