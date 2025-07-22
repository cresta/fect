# Simple FECT Evaluation Script

This directory contains a simple, easy-to-use script for evaluating claims on the FECT dataset using OpenAI models.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Run the evaluation:**
   ```bash
   python example.py
   ```

## Usage

### Basic Usage
```bash
python example.py
```
This runs the evaluation on the entire dataset using the default system prompt and model (gpt-4.1-mini).

### Custom System Prompt
```bash
python example.py --system-prompt "Your custom prompt here"
```

### Custom Model
```bash
python example.py --model "gpt-4o-2024-08-06"
```

### Combined Options
```bash
python example.py --model "gpt-4.1-2025-04-14" --system-prompt "Your custom prompt here"
```