import os
import functools
import pickle
import json
import hashlib
import copy

DISABLE_CACHE = os.getenv("DISABLE_CACHE", "false").lower() == "true"
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'judge_ablation_cache')

# Needed because these models are expensive to run
# Hence we don't want to lose any intermediate results in case of errors during inference over the entire dataset
def cache_api_call(prompt_mode: str, model_name_kwarg: str = 'model', prompt_arg_index: int = 0):
    """Simplified decorator to cache API call results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs = copy.deepcopy(kwargs)
            args = copy.deepcopy(args)
            
            current_prompt = args[prompt_arg_index]
            request_kwargs_dict = args[1] # Assuming second arg is request_kwargs for model_name
            model_name_from_kwargs = request_kwargs_dict.get(model_name_kwarg).split('/')[-1]
            
            serialized_prompt = json.dumps(current_prompt, sort_keys=True)
            file_hash = hashlib.md5(serialized_prompt.encode()).hexdigest()
            file_name = f"{model_name_from_kwargs}_{prompt_mode}_{file_hash}.pkl"
            file_path = os.path.join(CACHE_DIR, file_name)

            os.makedirs(CACHE_DIR, exist_ok=True)

            if not DISABLE_CACHE:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
            
            result = func(*args, **kwargs)

            with open(file_path, 'wb') as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator