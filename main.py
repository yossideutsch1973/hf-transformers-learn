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
    # Detailed prompt about sensor design
    prompt = """You are an expert in sensor design and electronics. 
    Explain in detail the easiest professional-grade sensor that can be built at home.
    Include:
    1. Type of sensor and what it measures
    2. Required components and approximate costs
    3. Basic assembly steps
    4. Expected accuracy/performance
    5. Potential applications
    
    Please provide a practical and detailed response:"""
    
    print("\nGenerating sensor design explanation...")
    
    # Generate text using the pipeline
    output = pipe(prompt, 
        max_new_tokens=500,  # Increased for detailed explanation
        num_beams=3,
        temperature=0.8,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )

    # Extract and clean the generated text
    generated_text = output[0]['generated_text'].split("Please provide a practical and detailed response:")[-1].strip()
    
    # Format output
    print("\nSensor Design Recommendation:")
    print(generated_text)

except Exception as e:
    print(f"\nError generating text: {str(e)}")
