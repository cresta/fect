#!/usr/bin/env python3
"""
Simple FECT dataset evaluation script.
This script provides a basic evaluation of claims using OpenAI models.
"""

import argparse
import os
import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report
from typing import List
from tqdm import tqdm
from pydantic import BaseModel

# Default system prompt (basic_no_ttc)
DEFAULT_SYSTEM_PROMPT = """Given a conversation and a claim about that conversation, determine if the claim is factual, i.e., supported by the conversation.

### Output Format as JSON:
answer: True or False (True if claim is factual; otherwise, False)."""

class FectResponse(BaseModel):
    answer: bool

def load_dataset() -> pd.DataFrame:
    """Load the FECT benchmark dataset."""
    dataset_path = "../../data/fect_benchmark.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    return pd.read_csv(dataset_path)

def run_inference(system_prompt: str, dataset: pd.DataFrame, model: str) -> List[bool]:
    """Run inference using OpenAI models."""
    client = OpenAI()
    
    results = []
    conversations = dataset['conversation'].tolist()
    claims = dataset['claim'].tolist()
    
    print(f"Running inference on {len(conversations)} examples using {model}...")
    
    for conversation, claim in tqdm(zip(conversations, claims), total=len(conversations)):
        user_prompt = f"""### Conversation ###
{conversation}

### Claim ###
{claim}"""
        
        try:
            response = client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=FectResponse,
                temperature=0.0
            )
            
            # Parse the response using the Pydantic model
            response = response.choices[0].message.parsed
            results.append(response.answer)
            
        except Exception as e:
            print(f"Error processing example: {e}")
            results.append(False)  # Default to False on error
    
    return results

def calculate_metrics(y_true: List[bool], y_pred: List[bool]) -> None:
    """Calculate and print precision, recall, and F1 scores."""
    print(classification_report(y_true, y_pred, target_names=['Non-factual', 'Factual']))

def main():
    parser = argparse.ArgumentParser(
        description="Simple FECT dataset evaluation using OpenAI models"
    )
    parser.add_argument(
        "--system-prompt", 
        type=str, 
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use for evaluation (default: basic_no_ttc prompt)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use for evaluation (default: gpt-4.1-mini)"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    print("Loading dataset...")
    dataset = load_dataset()
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Run inference
    predictions = run_inference(args.system_prompt, dataset, args.model)
    
    # Get ground truth labels
    ground_truth = dataset['claim_is_factual'].tolist()
    
    # Calculate and display metrics
    calculate_metrics(ground_truth, predictions)
    
    return 0

if __name__ == "__main__":
    exit(main()) 