#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ===== User-editable GPU override =====
# Set to an integer >= 1 to force number of GPUs (tensor parallel size).
# Leave as None to auto-detect based on CUDA_VISIBLE_DEVICES or torch.cuda.
TENSOR_PARALLEL_SIZE_OVERRIDE = 8
# GPU COUNT

def load_records(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    raise ValueError("Input JSON must be a list of records or contain a 'data' list")


def save_records(records: List[Dict[str, Any]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def build_prompt(title: str, abstract: str) -> str:
    """Build a Llama 3.1 instruct-style prompt for idea abstract generation"""
    definition = (
        "We define an 'idea abstract' as a version of the original abstract that "
        "deliberately omits specific results, implementation details, and performance metrics, "
        "while preserving the core problem statement, proposed approach, and claimed novelty."
    )
    instructions = (
        "Rewrite the following paper abstract into an idea abstract. "
        "Do not include any numbers, dataset sizes, epochs, hyperparameters, training or implementation details, "
        "or benchmark-specific scores. Keep the core problem, method, and novelty. Write concise, clear prose. "
        "Output only the rewritten idea abstract, without any preamble or explanation."
    )
    
    # Llama 3.1 Instruct format (no title in the user message)
    system_message = f"{definition}\n\n{instructions}"
    user_message = f"Abstract: {abstract}\n\nIdea abstract:"
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def generate_idea_abstracts(
    records: List[Dict[str, Any]],
    model_name: str,
    tensor_parallel_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    # Create tokenizer to get EOS id
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use Ray backend for multi-GPU to avoid multiprocessing fork issues
    distributed_backend = "ray" if tensor_parallel_size > 1 else None

    llm = LLM(
        model=model_name,
        tokenizer_mode="auto",
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        trust_remote_code=True,
        distributed_executor_backend=distributed_backend,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=[tokenizer.eos_token_id] if getattr(tokenizer, "eos_token_id", None) is not None else None,
    )

    prompts: List[str] = []
    for rec in records:
        title = rec.get("title") or ""
        abstract = rec.get("abstract") or ""
        abstract_placeholder = False
        if not abstract:
            abstract_placeholder = True
        else:
            low = abstract.strip().lower()
            if low in {"abstract", "abstract.tex"} or low.endswith(".tex") or "sections/" in low:
                abstract_placeholder = True
            elif len(abstract.strip()) < 20:
                abstract_placeholder = True

        if not abstract_placeholder:
            prompts.append(build_prompt(title, abstract))
        else:
            # Keep alignment with outputs (invalid/empty abstract -> empty idea abstract)
            prompts.append("")

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    results: List[str] = []
    for out in outputs:
        text = out.outputs[0].text if out.outputs else ""
        results.append(text.strip())
    return results


def detect_gpu_count() -> int:
    # Respect CUDA_VISIBLE_DEVICES if set
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        try:
            count = len([d for d in cuda_visible.split(",") if d.strip() != ""])
            return max(1, count)
        except Exception:
            pass
    # Avoid importing torch in the parent process to prevent CUDA init before spawn
    return 1


def compute_default_output_path(input_path: str) -> str:
    # Map .../retrieved_papers/<conf>/<year>/papers.json -> .../finished_data/<conf>/<year>/idea_abstracts.json
    norm = os.path.normpath(input_path)
    parts = norm.split(os.sep)
    try:
        idx = parts.index("retrieved_papers")
        base = os.sep.join(parts[:idx])
        conf = parts[idx + 1] if len(parts) > idx + 1 else "unknown_conf"
        year = parts[idx + 2] if len(parts) > idx + 2 else "unknown_year"
        return os.path.join(base, "finished_data", conf, year, "idea_abstracts_llama.json")
    except ValueError:
        # Fallback: put next to input under finished_data
        base = os.path.dirname(os.path.dirname(norm))
        return os.path.join(base, "finished_data", "idea_abstracts_llama.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 'idea abstracts' from {id, title, abstract, year} records using vLLM and Llama 3.1 8B Instruct."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON (list of {id, title, abstract, year} records)",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Path to output JSON. Default: finished_data/<conf>/<year>/idea_abstracts_llama.json",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name or path. Default: meta-llama/Llama-3.1-8B-Instruct",
    )
    # GPU count is now controlled in-code via TENSOR_PARALLEL_SIZE_OVERRIDE
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    # Use in-code override or auto-detect
    tps = TENSOR_PARALLEL_SIZE_OVERRIDE if TENSOR_PARALLEL_SIZE_OVERRIDE is not None else detect_gpu_count()
    if tps is None or tps < 1:
        raise ValueError("Number of GPUs must be >= 1. Set TENSOR_PARALLEL_SIZE_OVERRIDE or ensure CUDA is visible.")
    records = load_records(args.input)

    generations = generate_idea_abstracts(
        records=records,
        model_name=args.model,
        tensor_parallel_size=tps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_records: List[Dict[str, Any]] = []
    for rec, idea in zip(records, generations):
        merged = dict(rec)
        merged["idea_abstract"] = idea
        output_records.append(merged)

    out_path = args.output or compute_default_output_path(args.input)
    save_records(output_records, out_path)
    print(f"Wrote {len(output_records)} records to {out_path}")


if __name__ == "__main__":
    main()

