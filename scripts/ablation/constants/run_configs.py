from typing import List

PROMPTING_MODES = [
    "BASIC_WITH_TTC",
    "BASIC_NO_TTC",
    "3D_WITH_TTC",
    "3D_NO_TTC",
]

# Models designated as "reasoning models" for specific handling
OPENAI_REASONING_MODELS = [
    "o1-2024-12-17", 
    "o3-2025-04-16", 
    "o4-mini-2025-04-16",
]

# Other OpenAI models
OPENAI_STANDARD_MODELS = [
    "gpt-4o-2024-08-06", 
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
]  # Models not in the reasoning list

CLAUDE_BEDROCK_MODEL_IDS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
]

GEMINI_MODEL_IDS = [
    "gemini-2.5-flash-preview-04-17", 
    "gemini-2.5-pro-preview-05-06",
]

FIREWORKS_MODEL_IDS = [
    "accounts/fireworks/models/llama4-maverick-instruct-basic",
    "accounts/fireworks/models/deepseek-r1-basic",
]

HALLOUMI_MODEL_IDS = [
    "oumi-ai/HallOumi-8B-classifier",
    "oumi-ai/HallOumi-8B",
]

MODELS: List[str] = [
    *OPENAI_REASONING_MODELS,
    *OPENAI_STANDARD_MODELS,
    *CLAUDE_BEDROCK_MODEL_IDS,
    *GEMINI_MODEL_IDS,
    *FIREWORKS_MODEL_IDS,
    *HALLOUMI_MODEL_IDS,
]