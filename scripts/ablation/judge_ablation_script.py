import argparse
import traceback
import json
import os
import openai  # Added for OpenAI API
import boto3  # Added for AWS Bedrock
import re
import pandas as pd
import constants.prompts as prompts_constants
import constants.run_configs as run_configs
import constants.response_classes as response_classes
from google import genai
from tqdm import tqdm
from datetime import datetime
from typing import List, Any, Dict
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report
from inference_halloumi import inference_halloumi_8b_classifier, inference_halloumi_8b, get_claims_from_response
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import cache_api_call

SYSTEM_PROMPT = ""
XML_PROMPT_ADDITION = ""
RESPONSE_FORMAT = None
PROMPT_MODE = ""

def _run_inference_openai(model_name: str, dataset: pd.DataFrame) -> Any:
    conversations = dataset['conversation']
    short_answers = dataset['claim']
    
    client = openai.OpenAI()
    prompts = [
        [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': prompts_constants.FACTUALITY_USER_PROMPT.format(conversation=conversation, short_answer=short_answer)
            }
        ]
        for conversation, short_answer in zip(conversations, short_answers)
    ]

    request_kwargs = {
        'text_format': RESPONSE_FORMAT,
        'model': model_name,
        'store': False,
    }
    if model_name in run_configs.OPENAI_REASONING_MODELS:
        request_kwargs['reasoning'] = {'effort': 'high', 'summary': 'detailed'}
    if model_name in run_configs.OPENAI_STANDARD_MODELS:
        request_kwargs['temperature'] = 0.0
    
    @cache_api_call(prompt_mode=PROMPT_MODE, model_name_kwarg='model', prompt_arg_index=0)
    def _get_completion(prompt: List[Dict[str, str]], request_kwargs: Dict[str, Any]) -> Dict:        
        request_kwargs['input'] = prompt
        response = client.responses.parse(**request_kwargs)
        if hasattr(response.output[0], 'content') and hasattr(response.output[0].content[0], 'refusal'):
            print(
                f"    Model {request_kwargs['model']} refused to answer for safety reasons: {response.output[0].content[0].refusal}"
            )
            return {"error": "Model refusal", "details": response.output[0].content[0].refusal}
        
        reasoning = None
        for item in response.output:
            if hasattr(item, 'summary') and len(item.summary) > 0:
                reasoning = "\n\n".join([summary.text for summary in item.summary])
                break
        
        resp = {
            'parsed_output': response.output_parsed,
            'internal_reasoning': reasoning,
            'usage': response.usage.to_dict()
        }
        
        return resp
    
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for prompt in prompts:
            futures.append(executor.submit(_get_completion, prompt, request_kwargs))
        for future in tqdm(futures, total=len(futures), desc=f"Running completions for {model_name}"):
            results.append(future.result())
    return results


def _run_inference_halloumi(model_name: str, dataset: pd.DataFrame) -> Any:
    conversations = dataset['conversation']
    short_answers = dataset['claim']

    if 'classifier' in model_name:
        inference_results = inference_halloumi_8b_classifier(conversations, short_answers)
        results = [
            {
                "parsed_output": response_classes.FactualityResponseNoTTC(answer=result),
                "internal_reasoning": None,
                "usage": None,
            } for result in inference_results
        ]
    else:
        inference_results = inference_halloumi_8b(conversations, short_answers)
        results = []
        for result in inference_results:
            # The model's response is the last message of the result (a `Conversation` object).
            response = str(result.last_message().content)
            claims = get_claims_from_response(response)
            claims_list = []
            answer = True
            for claim in claims:
                claims_list.append(response_classes.Claim(claim=claim.claim_string, reasoning=claim.rationale, is_claim_verified=claim.supported))
                answer = answer and claim.supported
            results.append({
                "parsed_output": response_classes.FactualityResponse(claims=claims_list, reasoning="", answer=answer),
                "internal_reasoning": None,
                "usage": None,  
            })
    return results

def _run_inference_claude(model_name: str, dataset: pd.DataFrame) -> Any:
    conversations = dataset['conversation']
    short_answers = dataset['claim']
    
    boto3_session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        profile_name=os.getenv("AWS_PROFILE"),
        region_name=os.getenv("AWS_REGION"),
    )
    bedrock_runtime = boto3_session.client(service_name='bedrock-runtime')

    prompts = [
        [
            {
                'role': 'user',
                'content': prompts_constants.FACTUALITY_USER_PROMPT.format(conversation=conversation, short_answer=short_answer)
            }
        ]
        for conversation, short_answer in zip(conversations, short_answers)
    ]

    request_kwargs = {
        "modelId": model_name,
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT.replace("as JSON", "as XML") + "\n\nONLY generate XML. DO NOT generate any other text.\n\n" + XML_PROMPT_ADDITION,
        "max_tokens": 128000 if 'sonnet-4' not in model_name else 16000,
    }

    if "3-5" in model_name:
        request_kwargs['temperature'] = 0.0
    else:
        request_kwargs['thinking'] = {
            "type": "enabled",
            "budget_tokens": 1024
        }

    @cache_api_call(prompt_mode=PROMPT_MODE, model_name_kwarg='modelId', prompt_arg_index=0)
    def _get_completion(prompt: List[Dict[str, str]], request_kwargs: Dict[str, Any]) -> Dict:
        request_kwargs['messages'] = prompt
        request_kwargs.pop('modelId')
        response = bedrock_runtime.invoke_model(body=json.dumps(request_kwargs), modelId=model_name)
        response_body = json.loads(response.get('body').read())
        content = response_body['content']
        reasoning = None
        parsed_output = None
        for item in content:
            if item['type'] == 'thinking':
                reasoning = item['thinking']
            elif item['type'] == 'text':
                try:
                    text = item['text']
                    text = text.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
                    text = text.replace('&', '&amp;').strip()
                    text = text[text.index('<factuality_response>'): text.index('</factuality_response>') + len('</factuality_response>') + 1]
                    parsed_output = RESPONSE_FORMAT.from_xml(text)
                except Exception as e:
                    print(f"Error parsing XML response from Claude: {e}")
                    print(f"Raw response content: {item['text']}")
                    raise e
            else:
                raise ValueError(f"Unknown item type: {item['type']}")
        resp = {
            'parsed_output': parsed_output,
            'internal_reasoning': reasoning,
            'usage': response_body['usage']
        }
        
        return resp
    
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=10 if 'sonnet-4' not in model_name else 1) as executor:
        for prompt in prompts:
            futures.append(executor.submit(_get_completion, prompt, request_kwargs))
        for future in tqdm(futures, total=len(futures), desc=f"Running completions for {model_name}"):
            results.append(future.result())
    return results



def _run_inference_gemini(model_name: str, dataset: pd.DataFrame) -> Any:
    conversations = dataset['conversation']
    short_answers = dataset['claim']
    prompts = [
        SYSTEM_PROMPT + "\n\n" + prompts_constants.FACTUALITY_USER_PROMPT.format(conversation=conversation, short_answer=short_answer)
        for conversation, short_answer in zip(conversations, short_answers)
    ]
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    request_kwargs = {
        "model": model_name,
        "config": {
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_FORMAT,
            'thinking_config': {"thinking_budget": 24576, 'include_thoughts': True}
        }
    }

    @cache_api_call(prompt_mode=PROMPT_MODE, model_name_kwarg='model', prompt_arg_index=0)
    def _get_completion(prompt: str, request_kwargs: Dict[str, Any]) -> Dict:
        request_kwargs['contents'] = prompt
        response = client.models.generate_content(**request_kwargs)
        return {
            "parsed_output": response.parsed,
            "usage": response.usage_metadata.dict()
        }
    
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for prompt in prompts:
            futures.append(executor.submit(_get_completion, prompt, request_kwargs))
        for future in tqdm(futures, total=len(futures), desc=f"Running completions for {model_name}"):
            results.append(future.result())
    return results


def _run_inference_fireworks(model_name: str, dataset: pd.DataFrame) -> Any:
    conversations = dataset['conversation']
    short_answers = dataset['claim']

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv("FIREWORKS_API_KEY"),
    )
    prompts = [
        [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': prompts_constants.FACTUALITY_USER_PROMPT.format(conversation=conversation, short_answer=short_answer)
            }
        ]
        for conversation, short_answer in zip(conversations, short_answers)
    ]
    request_kwargs = {
        'response_format': {"type": "json_object", "schema": RESPONSE_FORMAT.model_json_schema()},
        'model': model_name,
        'temperature': 0.0,
        'max_tokens': 32000,
    }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    @cache_api_call(prompt_mode=PROMPT_MODE, model_name_kwarg='model', prompt_arg_index=0)
    def _get_completion(prompt: List[Dict[str, str]], request_kwargs: Dict[str, Any]) -> Dict:        
        request_kwargs['messages'] = prompt
        response = client.chat.completions.create(**request_kwargs)
        response_content = response.choices[0].message.content
        internal_reasoning = None
        if "deepseek-r1" in model_name:
            reasoning_match = re.search(r"<think>(.*?)</think>", response_content, re.DOTALL)
            if reasoning_match:
                internal_reasoning = reasoning_match.group(1).strip()
            json_match = re.search(r"</think>\s*(\{.*\})", response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(1).strip()
            else:
                raise ValueError(f"No JSON match found in response for {model_name}: {response_content}")
            
        resp = {
            'parsed_output': RESPONSE_FORMAT.model_validate_json(response_content),
            'internal_reasoning': internal_reasoning,
            'usage': response.usage.to_dict()
        }
        
        return resp
    
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for prompt in prompts:
            futures.append(executor.submit(_get_completion, prompt, request_kwargs))
        for future in tqdm(futures, total=len(futures), desc=f"Running completions for {model_name}"):
            results.append(future.result())
    return results


# --- Main Inference Dispatcher ---
def run_inference(model_name: str, dataset: pd.DataFrame) -> Any:
    """
    Runs inference for a given model and prompt by dispatching to a model-specific function.
    """
    print(f"  Dispatching inference for model '{model_name}' with {len(dataset)} prompts...")
    if model_name in run_configs.OPENAI_REASONING_MODELS or model_name in run_configs.OPENAI_STANDARD_MODELS:
        return _run_inference_openai(model_name, dataset)
    elif model_name in run_configs.CLAUDE_BEDROCK_MODEL_IDS:
        return _run_inference_claude(model_name=model_name, dataset=dataset)
    elif model_name in run_configs.FIREWORKS_MODEL_IDS:  # Added Llama via Fireworks
        return _run_inference_fireworks(model_name=model_name, dataset=dataset)
    elif "oumi" in model_name:
        return _run_inference_halloumi(model_name=model_name, dataset=dataset)
    elif model_name in run_configs.GEMINI_MODEL_IDS:
        return _run_inference_gemini(model_name=model_name, dataset=dataset)
    else:
        raise ValueError(
            f"Unknown model for inference: {model_name}. No specific inference function defined.")

def load_fect_dataset() -> pd.DataFrame:
    """
    Loads the FECT dataset.
    """
    df = pd.read_csv("../data/fect_benchmark.csv")
    return df


def process_output(inference_output: List[Dict[str, Any]]) -> pd.DataFrame:
    """Processes the inference output and returns a DataFrame."""
    for output in inference_output:
        output['answer'] = output['parsed_output'].answer
        if hasattr(output['parsed_output'], 'claims'):
            output['claims'] = [claim.model_dump() for claim in output['parsed_output'].claims]
        else:
            output['claims'] = []
        if hasattr(output['parsed_output'], 'reasoning'):
            output['ttc_reasoning'] = output['parsed_output'].reasoning
        else:
            output['ttc_reasoning'] = None
        output.pop('parsed_output')
    return pd.DataFrame(inference_output)


# --- Main Ablation Script ---
def run_ablation_study(models: List[str], prompt_modes: List[str]):
    """
    Runs the ablation study across all specified dimensions.
    """
    assert set(models).issubset(set(run_configs.MODELS)), f"Models must be a subset of {run_configs.MODELS}"
    assert set(prompt_modes).issubset(set(run_configs.PROMPTING_MODES)), f"Prompt modes must be a subset of {run_configs.PROMPTING_MODES}"
    models = set(models)
    prompt_modes = set(prompt_modes)

    all_results = []
    script_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Running ablation study for models: {models}, prompt modes: {prompt_modes}")
    for model_name in models:
        for prompting_mode in prompt_modes:
            global SYSTEM_PROMPT
            global XML_PROMPT_ADDITION
            global RESPONSE_FORMAT
            global PROMPT_MODE

            PROMPT_MODE = prompting_mode

            if prompting_mode == "BASIC_WITH_TTC":
                RESPONSE_FORMAT = response_classes.FactualityResponseBasic if 'claude' not in model_name else response_classes.FactualityResponseBasicXML
                SYSTEM_PROMPT = prompts_constants.FACTUALITY_SYSTEM_PROMPT_BASIC_WITH_TTC
                XML_PROMPT_ADDITION = prompts_constants.XML_OUTPUT_BASIC_SYSTEM_PROMPT_WITH_TTC_ADDITION
            elif prompting_mode == "BASIC_NO_TTC":
                RESPONSE_FORMAT = response_classes.FactualityResponseNoTTC if 'claude' not in model_name else response_classes.FactualityResponseXMLNoTTC
                SYSTEM_PROMPT = prompts_constants.FACTUALITY_SYSTEM_PROMPT_BASIC_NO_TTC
                XML_PROMPT_ADDITION = prompts_constants.XML_OUTPUT_BASIC_SYSTEM_PROMPT_NO_TTC_ADDITION
            elif prompting_mode == "3D_NO_TTC":
                RESPONSE_FORMAT = response_classes.FactualityResponseNoTTC if 'claude' not in model_name else response_classes.FactualityResponseXMLNoTTC
                SYSTEM_PROMPT = prompts_constants.FACTUALITY_SYSTEM_PROMPT_3D_NO_TTC
                XML_PROMPT_ADDITION = prompts_constants.XML_OUTPUT_SYSTEM_PROMPT_NO_TTC_ADDITION
            else: # 3D_WITH_TTC
                RESPONSE_FORMAT = response_classes.FactualityResponse if 'claude' not in model_name else response_classes.FactualityResponseXML
                SYSTEM_PROMPT = prompts_constants.FACTUALITY_SYSTEM_PROMPT_3D_WITH_TTC
                XML_PROMPT_ADDITION = prompts_constants.XML_OUTPUT_SYSTEM_PROMPT_WITH_TTC_ADDITION

            print(
                f"\nProcessing: Model='{model_name}', Prompting Mode='{prompting_mode}'"
            )

            # 1. Load dataset
            dataset: pd.DataFrame = load_fect_dataset()

            # 3. Iterate through dataset examples for this configuration
            try:
                inference_output = run_inference(model_name=model_name, dataset=dataset)
            except Exception as e:
                print(f"Error running inference for {model_name} on prompting mode {prompting_mode}: {e}")
                with open(f"fect_benchmark_error_{script_timestamp}.txt", "a") as f:
                    f.write(f"Error running inference for {model_name} on prompting mode {prompting_mode}: {traceback.format_exc()}\n\n")
                continue
            inference_output_df = process_output(inference_output)
            dataset = pd.concat([dataset, inference_output_df], axis=1)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tag = f"{model_name.split('/')[-1]}_{prompting_mode}_{timestamp}"
            dataset.to_csv(f"fect_benchmark_{tag}.csv", index=False)
            print(f"Classification report for {model_name}:")
            print(classification_report(dataset['claim_is_factual'], dataset['answer']))
            report = classification_report(dataset['claim_is_factual'], dataset['answer'], output_dict=True, zero_division=0)
            with open(f"fect_benchmark_results_{tag}.json", "w") as f:
                json.dump(report, f, indent=4)

            all_results.append({
                'model_name': model_name.split('/')[-1],
                'prompting_mode': prompting_mode,
                'timestamp': timestamp,
                'hallucination_precision': report['False']['precision'],
                'hallucination_recall': report['False']['recall'],
                'hallucination_f1_score': report['False']['f1-score'],
                'report': json.dumps(report)
            })
            
            if 'oumi' in model_name.lower():
                break

    pd.DataFrame(all_results).to_csv(f"fect_benchmark_results_all_{script_timestamp}.csv", index=False)
    print("\n--- Ablation Study Complete (check logs for errors) ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM Judge Ablation Study")
    parser.add_argument(
        "--models",
        nargs="*",
        default=run_configs.MODELS,
        help="List of model names to use (default: all models from run_configs)",
    )
    parser.add_argument(
        "--prompt_modes",
        nargs="*",
        default=run_configs.PROMPTING_MODES,
        help="List of prompting modes to use (default: all modes from run_configs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_study(models=args.models, prompt_modes=args.prompt_modes)
