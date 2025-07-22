import gc
import torch
import nltk
import contextlib
from typing import Dict
from oumi import infer as infer_oumi
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass, field
from oumi.core.configs import InferenceConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from typing import Any, List
from tqdm import tqdm
from utils import cache_api_call
nltk.download("punkt_tab")


def inference_halloumi_8b_classifier(conversations: List[str], short_answers: List[str]) -> Any:
    model = AutoModelForSequenceClassification.from_pretrained("oumi-ai/HallOumi-8B-classifier",
                                                               torch_dtype=torch.float16,
                                                               device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("oumi-ai/HallOumi-8B-classifier")

    PROMPT_TEMPLATE = "<context>\n{context}\n</context>\n\n<claims>\n{claim}\n</claims>"

    @cache_api_call(prompt_mode="halloumi", model_name_kwarg='model_name', prompt_arg_index=0)
    def _infer(prompt: str, kwargs: Dict[str, Any]):
        debug = kwargs.get('debug', False)
        inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        if debug:
            print(f"INPUTS: {inputs}")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.to('cpu')
        if debug:
            print(f"Logits shape: {logits.shape}")
            print(f"Logits: {logits}")
        probabilities = softmax(logits.numpy(), axis=-1)
        prediction_inx = probabilities.argmax()
        if debug:
            print(f"Prediction Index: {prediction_inx}")
        prediction_label = False if prediction_inx == 1 else True
        return prediction_label, probabilities[0][prediction_inx]

    results = []
    for conversation, short_answer in tqdm(zip(conversations, short_answers)):
        prompt = PROMPT_TEMPLATE.format(context=conversation, claim=short_answer)
        label, _ = _infer(prompt, {"model_name": "oumi-ai/HallOumi-8B-classifier"})
        results.append(label)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def inference_halloumi_8b(conversations: List[str], short_answers: List[str]) -> Any:
    local_config_str = """
model:
    model_name: "oumi-ai/HallOumi-8B"
    trust_remote_code: true
    torch_dtype_str: "float16"
    model_max_length: 17000

generation:
    max_new_tokens: 512
    temperature: 0.0

engine: VLLM  # Set to VLLM, if you have a CUDA-compatible GPU.
"""

    inference_config = InferenceConfig.from_str(local_config_str)
    prompts = [
        create_prompt(context=conversation,
                      request="Make one or more claims about information in the documents.",
                      response=short_answer)
        for conversation, short_answer in zip(conversations, short_answers)
    ]

    inference_results = infer_oumi(
        config=inference_config,
        inputs=prompts,
    )

    return inference_results


def create_prompt(context: str, request: str, response: str) -> str:
    """Generates a prompt for the generative HallOumi model."""
    def _split_into_sentences(text: str) -> list[str]:
        sentences = sent_tokenize(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _annotate_sentences(sentences: list[str], annotation_char: str) -> str:
        annotated_sentences = []
        for idx, sentence in enumerate(sentences, start=1):
            annotated_sentences.append(
                f"<|{annotation_char}{idx}|><{sentence}><end||{annotation_char}>")
        return "".join(annotated_sentences)

    # Context: Split it into sentences and annotate them.
    context_sentences = _split_into_sentences(context)
    annotated_context_sentences = _annotate_sentences(context_sentences, "s")
    annotated_context = f"<|context|>{annotated_context_sentences}<end||context>"

    # Request: Annotate the request.
    annotated_request = f"<|request|><{request.strip()}><end||request>"

    # Response: Split it into sentences and annotate them.
    response_sentences = _split_into_sentences(response)
    annotated_response_sentences = _annotate_sentences(response_sentences, "r")
    annotated_response = f"<|response|>{annotated_response_sentences}<end||response>"

    # Combine all parts into the final prompt.
    return f"{annotated_context}{annotated_request}{annotated_response}"


@dataclass
class Claim:
    claim_id: int = -1
    claim_string: str = ""
    subclaims: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    rationale: str = ""
    supported: bool = True


def get_claims_from_response(response: str) -> list[Claim]:
    """Extracts claims from the response string."""
    def _get_claim_id_from_subsegment(subsegment: str) -> int:
        claim_id_part = subsegment.split("|")[1]
        claim_id_no_r = claim_id_part.lstrip("r")
        return int(claim_id_no_r)

    def _get_claim_citations_from_subsegment(subsegment: str) -> list[int]:
        citation_segments = subsegment.split(",")
        citations = []
        for citation_segment in citation_segments:
            citation = citation_segment.replace("|", "").replace("s", "").strip()
            if "-" in citation:
                start, end = map(int, citation.split("-"))
                citations.extend(range(start, end + 1))
            elif "to" in citation:
                start, end = map(int, citation.split("to"))
                citations.extend(range(start, end + 1))
            else:
                with contextlib.suppress(ValueError):
                    citation_int = int(citation)
                    citations.append(citation_int)
        return citations

    def _get_claim_from_segment(segment: str) -> Claim:
        claim_segments = segment.split("><")
        claim = Claim()
        claim.claim_id = _get_claim_id_from_subsegment(claim_segments[0])
        claim.claim_string = claim_segments[1]

        subclaims = []
        claim_progress_index = 3  # start parsing subclaims from index 3
        for i in range(claim_progress_index, len(claim_segments)):
            subsegment = claim_segments[i]
            if subsegment.startswith("end||subclaims"):
                claim_progress_index = i + 1
                break
            subclaims.append(subsegment)

        citation_index = -1
        rationale_index = -1
        label_index = -1

        for i in range(claim_progress_index, len(claim_segments)):
            subsegment = claim_segments[i]
            if subsegment.startswith("|cite|"):
                citation_index = i + 1
            elif subsegment.startswith("|explain|"):
                rationale_index = i + 1
            elif subsegment.startswith("|supported|") or subsegment.startswith("|unsupported|"):
                label_index = i

        claim.subclaims = subclaims
        claim.citations = (_get_claim_citations_from_subsegment(claim_segments[citation_index])
                           if citation_index != -1 else [])
        claim.rationale = (claim_segments[rationale_index] if rationale_index != -1 else "")
        claim.supported = (claim_segments[label_index].startswith("|supported|")
                           if label_index != -1 else True)
        return claim

    segments = response.split("<end||r>")
    claims = []
    for segment in segments:
        if segment.strip():
            claim = _get_claim_from_segment(segment)
            claims.append(claim)
    return claims
