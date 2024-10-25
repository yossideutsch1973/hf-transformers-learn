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

# Initialize the pipeline with a more technical model
pipe = pipeline("text-generation",
    model="databricks/dolly-v2-3b",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

try:
    # Detailed prompt about sensor design
    prompt = """You are a professional electronics engineer specializing in sensor design. 
    Describe how to build a high-quality temperature and humidity sensor system at home.
    
    Structure your response exactly like this:
    SENSOR TYPE:
    - Digital temperature and humidity sensor
    - Measurement range and capabilities
    
    COMPONENTS NEEDED:
    - List each component with approximate cost
    - Include any required tools
    
    ASSEMBLY STEPS:
    1. Detailed step-by-step instructions
    2. Include wiring diagram description
    3. Basic programming requirements
    
    PERFORMANCE SPECS:
    - Temperature accuracy
    - Humidity accuracy
    - Response time
    
    APPLICATIONS:
    - List practical uses
    - Potential projects
    
    Provide specific details and keep it practical for a home builder:"""
    
    print("\nGenerating sensor design explanation...")
    
    # Generate text using the pipeline
    output = pipe(prompt, 
        max_new_tokens=800,  # Longer response for detailed instructions
        num_beams=4,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3
    )

    # Extract and clean the generated text
    generated_text = output[0]['generated_text'].split("Please provide a practical and detailed response:")[-1].strip()
    
    # Format output
    print("\nSensor Design Recommendation:")
    print(generated_text)

except Exception as e:
    print(f"\nError generating text: {str(e)}")
