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
    # Detailed prompt with structure
    prompt = """Write a nature haiku following this exact structure:
Line 1: 5 syllables about a rabbit
Line 2: 7 syllables describing its action
Line 3: 5 syllables with nature imagery

Example:
Soft rabbit hiding
Beneath the garden flowers
Morning dew sparkles

Now write a new one:"""
    
    print("\nGenerating haiku...")
    
    # Generate text using the pipeline
    output = pipe(prompt, 
        max_new_tokens=50,
        num_beams=4,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )

    # Extract and clean the generated text
    generated_text = output[0]['generated_text'].split("Now write a new one:")[-1].strip()
    
    # Format output
    print("\nGenerated Haiku:")
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    if len(lines) >= 3:
        for line in lines[:3]:
            print(line)
    else:
        print("Failed to generate complete haiku structure")

except Exception as e:
    print(f"\nError generating text: {str(e)}")
