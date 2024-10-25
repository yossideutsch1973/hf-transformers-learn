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

model_id = "microsoft/git-base"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Write a haiku (3 lines of 5-7-5 syllables) about this image. Focus on nature and seasonal elements:"
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
    add_special_tokens=True,
    max_length=512,
    padding=True,
    truncation=True
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=30,
    num_beams=10,
    temperature=0.9,
    no_repeat_ngram_size=3,
    do_sample=True,
    top_k=50,
    top_p=0.95
)
generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
print("\nGenerated Haiku:")
print(generated_text)
