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

# Initialize the pipeline with Llama 3.2 3B model
pipe = pipeline("text-generation",
    model="meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

try:
    # Detailed prompt about sensor design
    prompt = """You are a professional electronics engineer specializing in sensor systems. Create a detailed guide specifically for the BME280 temperature, humidity and pressure sensor.

    IMPORTANT: Your response must EXACTLY follow this format with all sections. Any deviation will be rejected.

    SENSOR TYPE:
    • BME280 specifications
    • Temperature range: -40°C to +85°C
    • Humidity range: 0-100% RH
    • Pressure range: 300-1100 hPa
    • Key features: I2C/SPI digital interface, high accuracy

    COMPONENTS NEEDED:
    • BME280 sensor module ($10-15)
    • Microcontroller (Arduino/ESP32) ($5-20) 
    • Jumper wires ($2-5)
    • Breadboard ($5)
    • USB cable ($2-3)
    • Optional: Display, case

    ASSEMBLY STEPS:
    1. Wire BME280 to microcontroller:
       - VCC to 3.3V
       - GND to GND
       - SCL to SCL/Clock
       - SDA to SDA/Data
    2. Install required libraries
    3. Upload example code
    4. Test and verify readings

    PERFORMANCE SPECIFICATIONS:
    • Temperature accuracy: ±0.5°C
    • Humidity accuracy: ±3% RH
    • Pressure accuracy: ±1 hPa
    • Response time: 1 second
    • Operating voltage: 3.3V
    • Current draw: 3.6μA at 1Hz sampling

    PRACTICAL APPLICATIONS:
    • Weather station monitoring
    • Smart home climate control
    • Indoor air quality tracking
    • Industrial process monitoring
    • Agricultural monitoring
    • IoT environmental sensing

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
        max_new_tokens=3000,
        num_beams=5,
        temperature=0.1,  # Even lower for more deterministic output
        do_sample=True, 
        top_k=10,  # More restricted sampling
        top_p=0.3,  # More focused output
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,  # Increased to avoid repetition
        length_penalty=2.0  # Increased to encourage complete sections
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
        
    # Validate and format the output
    if not all(section in generated_text for section in required_sections):
        # If sections are missing, try to regenerate with a stronger format reminder
        print("\nRetrying generation with stronger format requirements...")
        generated_text = pipe(prompt + "\n\nNOTE: ALL sections listed above MUST be included in your response.",
            max_new_tokens=3000,
            temperature=0.2,  # Even lower temperature for more focused output
            num_beams=5,
            do_sample=True,
            top_k=30,
            top_p=0.7,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            length_penalty=1.5
        )[0]['generated_text']
        
        if "SENSOR TYPE:" in generated_text:
            generated_text = generated_text[generated_text.index("SENSOR TYPE:"):]
    
    # Add common troubleshooting tips
    generated_text += "\n\nCOMMON TROUBLESHOOTING TIPS:"
    generated_text += "\n1. No readings or incorrect values:"
    generated_text += "\n   - Check power supply voltage (3.3V required)"
    generated_text += "\n   - Verify I2C/SPI connections and addresses"
    generated_text += "\n   - Ensure proper grounding"
    generated_text += "\n2. Unstable readings:"
    generated_text += "\n   - Add decoupling capacitors"
    generated_text += "\n   - Shield from EMI/RFI interference"
    generated_text += "\n   - Check for proper ventilation"
    generated_text += "\n3. Communication errors:"
    generated_text += "\n   - Verify bus speed settings"
    generated_text += "\n   - Check pull-up resistors"
    generated_text += "\n   - Update firmware/libraries"

    # Add specific use cases
    generated_text += "\n\nRECOMMENDED APPLICATIONS:"
    generated_text += "\n1. Smart Home:"
    generated_text += "\n   - HVAC control and optimization"
    generated_text += "\n   - Indoor air quality monitoring"
    generated_text += "\n   - Smart thermostat integration"
    generated_text += "\n2. Weather Monitoring:"
    generated_text += "\n   - Personal weather stations"
    generated_text += "\n   - Agricultural monitoring"
    generated_text += "\n   - Data logging systems"
    generated_text += "\n3. Industrial:"
    generated_text += "\n   - Clean room monitoring"
    generated_text += "\n   - Process control"
    generated_text += "\n   - Environmental compliance"

    # Add reference links
    generated_text += "\n\nUSEFUL REFERENCES:"
    generated_text += "\nDatasheet: https://www.bosch-sensortec.com/media/bosch_sensortec/downloads/datasheets/bst-bme280-ds002.pdf"
    generated_text += "\nTutorial: https://learn.adafruit.com/adafruit-bme280-humidity-barometric-pressure-temperature-sensor-breakout"
    generated_text += "\nSource Code: https://github.com/adafruit/Adafruit_BME280_Library"
    
    # Format and print the output
    print("\nSensor Design Recommendation:")
    print(generated_text)

except Exception as e:
    print(f"\nError generating text: {str(e)}")
