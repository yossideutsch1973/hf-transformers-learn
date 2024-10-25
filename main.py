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
    prompt = """You are a professional electronics engineer. Create a detailed guide for building a temperature and humidity sensor system at home.

    Your response MUST follow this EXACT format:

    SENSOR TYPE:
    • Specify exact sensor model (e.g. DHT22, BME280, etc.)
    • List full measurement ranges for temp/humidity
    • State key features (digital/analog, accuracy class)

    COMPONENTS NEEDED:
    • Full parts list with exact model numbers
    • Estimated cost per component in USD
    • Required tools and equipment
    • Any optional components for enhancements

    ASSEMBLY STEPS:
    1. Detailed wiring instructions (pin-by-pin)
    2. Power supply requirements
    3. Microcontroller setup (if needed)
    4. Code examples in Python/Arduino
    5. Common wiring mistakes to avoid

    PERFORMANCE SPECIFICATIONS:
    • Temperature accuracy (±°C)
    • Humidity accuracy (±%RH) 
    • Response time (seconds)
    • Operating voltage range
    • Power consumption
    • Expected lifetime

    PRACTICAL APPLICATIONS:
    • Home automation examples
    • Weather monitoring
    • Industrial use cases
    • Integration ideas

    TROUBLESHOOTING:
    • Common issues and solutions
    • Calibration procedure
    • Maintenance requirements

    Use bullet points and numbering. Include specific part numbers and values. Keep it practical for DIY builders."""
    
    print("\nGenerating sensor design explanation...")
    
    # Generate text using the pipeline with optimized parameters
    output = pipe(prompt,
        max_new_tokens=3000,  # Increased further for more detailed response
        num_beams=5,
        temperature=0.3,  # Reduced more for higher coherence
        do_sample=True,
        top_k=30,
        top_p=0.7,  # Reduced for more focused output
        no_repeat_ngram_size=3,
        repetition_penalty=1.8,  # Increased to avoid repetition
        length_penalty=1.5  # Added to encourage longer, more detailed responses
    )

    # Extract and clean the generated text
    generated_text = output[0]['generated_text']
    
    # Remove any prefix before the actual response
    if "SENSOR TYPE:" in generated_text:
        generated_text = generated_text[generated_text.index("SENSOR TYPE:"):]
    
    # Ensure all required sections are present
    required_sections = [
        "SENSOR TYPE:",
        "COMPONENTS NEEDED:",
        "ASSEMBLY STEPS:",
        "PERFORMANCE SPECIFICATIONS:",
        "PRACTICAL APPLICATIONS:",
        "TROUBLESHOOTING:"
    ]
    
    missing_sections = [section for section in required_sections if section not in generated_text]
    if missing_sections:
        print("\nWarning: Generated response is missing these sections:", ", ".join(missing_sections))
        
    # Format and print the output
    print("\nSensor Design Recommendation:")
    print(generated_text)

except Exception as e:
    print(f"\nError generating text: {str(e)}")
