# Hugging Face Transformers Learning Project

This project demonstrates using the Hugging Face Transformers library to generate detailed technical documentation using Large Language Models (LLMs).

## Features

- Uses Llama model for text generation
- Generates structured technical documentation
- Handles authentication with Hugging Face Hub
- Includes error handling and retry logic

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yossideutsch1973/hf-transformers-learn.git
cd hf-transformers-learn
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Hugging Face token:
- Get your token from https://huggingface.co/settings/tokens
- Set it as an environment variable:
```bash
export HF_TOKEN=your_token_here
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Authenticate with Hugging Face Hub
2. Load the Llama model
3. Generate detailed technical documentation based on the prompt
4. Format and display the results

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- Hugging Face account with appropriate model access
