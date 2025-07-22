# FECT Ablation Study

This directory contains the comprehensive evaluation script for running ablation studies on the FECT benchmark across multiple models and prompting strategies.

## Setup Instructions

### 1. Environment Setup
Requires `python >= 3.10`. Create a Python virtual environment and install dependencies:

```bash
# Navigate to the scripts directory
cd fect/scripts/ablation/

# Create virtual environment [OPTIONAL; RECOMMENDED]
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

The evaluation script supports multiple LLM providers. Set up API keys for the models you want to evaluate:

#### OpenAI Models
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Claude Models (via AWS Bedrock)
```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_SESSION_TOKEN="your-aws-session-token"  # Optional
export AWS_PROFILE="your-aws-profile"              # Optional
export AWS_REGION="us-east-1"                      # Or your preferred region
```

#### Google Gemini Models
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

#### Fireworks Models
For certain open weights / open source models like Deepseek and Llama we use the [Fireworks](https://fireworks.ai/) platform for inference.
```bash
export FIREWORKS_API_KEY="your-fireworks-api-key"
```

#### Optional: Disable Caching
```bash
export DISABLE_CACHE="true"  # Set to disable result caching
```

### 3. GPU Setup (for HallOumi models)

HallOumi models require GPU support. Ensure you have:
- CUDA-compatible GPU
- Appropriate CUDA drivers
- PyTorch with CUDA support

The requirements.txt includes GPU-enabled versions of PyTorch and related libraries.

## Usage

### Running the Full Benchmark

Execute the main evaluation script:

```bash
python judge_ablation_script.py
```

### Custom Model Selection

Evaluate specific models:

```bash
python judge_ablation_script.py --models gpt-4o-2024-08-06 anthropic.claude-3-5-sonnet-20240620-v1:0
```

### Custom Prompting Modes

Evaluate specific prompting strategies:

```bash
python judge_ablation_script.py --prompt_modes BASIC_WITH_TTC 3D_WITH_TTC
```

### Available Models

The framework supports the following model families:

- **OpenAI**: GPT-4o, GPT-4o-mini, O1, O3, GPT-4.1 series
- **Claude**: Claude-3.5-Sonnet, Claude-3.7-Sonnet, Claude-Sonnet-4
- **Gemini**: Gemini-2.5-flash, Gemini-2.5-pro
- **Fireworks**: Llama4-Maverick, DeepSeek-R1
- **HallOumi**: HallOumi-8B, HallOumi-8B-classifier
    - For `Halloumi-8B`, we recommend a GPU with atleast 24G memory.

### Prompting Modes

Four evaluation modes are available:

1. **BASIC_WITH_TTC**: Basic prompting with test time compute
2. **BASIC_NO_TTC**: Basic prompting without test time compute
3. **3D_WITH_TTC**: 3D claim analysis with test time compute
4. **3D_NO_TTC**: 3D claim analysis without test time compute

## Output

The script generates several output files:

- `fect_benchmark_{model}_{mode}_{timestamp}.csv`: Detailed results for each model/mode combination
- `fect_benchmark_results_{model}_{mode}_{timestamp}.json`: Classification metrics
- `fect_benchmark_results_all_{timestamp}.csv`: Aggregated results across all runs
- `fect_benchmark_error_{timestamp}.txt`: Error logs (if any)

## Evaluation Metrics

The framework reports standard classification metrics:
- Precision, Recall, F1-score for factual and non-factual claims
- Macro and weighted averages
- Confusion matrices 