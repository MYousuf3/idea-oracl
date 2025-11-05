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


def load_records(input_path: str, ranks_path: str) -> List[Dict[str, Any]]:
    # Load backtest paper list
    with open(ranks_path, "r") as f:
        backtest_papers = json.load(f)
    backtest_titles = {p["title"]: True for p in backtest_papers}
    
    # Load and filter papers
    with open(input_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    elif not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records or contain a 'data' list")
    
    # Only keep papers in the backtest set
    return [p for p in data if p.get("title", "") in backtest_titles]


def save_records(records: List[Dict[str, Any]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def build_prompt(title: str, abstract: str) -> str:
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

    system_message = f"{definition}\n\n{instructions}"
    user_message = f"Abstract: {abstract}\n\nIdea abstract:"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def build_prompt_proposal(title: str, content: str) -> str:
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
        "Answer with the following format:"

        "1. Title: A concise statement of the main research question to be used as the paper title."
        "2. Problem Statement: Clearly define the problem your research intends to address. Explain clearly why this problem is interesting and important."
        "3. Motivation: Explain why existing methods are not good enough to solve the problem, and explain the inspiration behind the new proposed method. You should also motivate why the proposed method would work better than existing baselines on the problem."
        "4. Proposed Method: Explain how the proposed method works, describe all the essential steps."
    )

    system_message = f"{definition}\n\n{instructions}"
    user_message = f"Paper content: {content}\n\nProposal:"

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    idea_prompts: List[str] = []
    proposal_prompts: List[str] = []
    for rec in records:
        title = rec.get("title") or ""
        abstract = rec.get("abstract") or ""
        content = rec.get("content") or ""

        # Determine if abstract is usable
        abstract_placeholder = False
        if not abstract:
            abstract_placeholder = True
        else:
            low = abstract.strip().lower()
            if low in {"abstract", "abstract.tex"} or low.endswith(".tex") or "sections/" in low:
                abstract_placeholder = True
            elif len(abstract.strip()) < 20:
                abstract_placeholder = True

        # For proposal prefer full content, else fallback to abstract
        proposal_input = content if (content and len(content.strip()) >= 20) else abstract
        proposal_placeholder = False
        if not proposal_input:
            proposal_placeholder = True
        else:
            lowp = proposal_input.strip().lower()
            if lowp in {"abstract", "abstract.tex"} or lowp.endswith(".tex") or "sections/" in lowp:
                proposal_placeholder = True
            elif len(proposal_input.strip()) < 20:
                proposal_placeholder = True

        if not abstract_placeholder:
            idea_prompts.append(build_prompt(title, abstract))
        else:
            idea_prompts.append("")

        if not proposal_placeholder:
            proposal_prompts.append(build_prompt_proposal(title, proposal_input))
        else:
            proposal_prompts.append("")

    idea_outputs = llm.generate(idea_prompts, sampling_params=sampling_params)
    idea_results: List[str] = []
    for out in idea_outputs:
        text = out.outputs[0].text if out.outputs else ""
        idea_results.append(text.strip())

    proposal_outputs = llm.generate(proposal_prompts, sampling_params=sampling_params)
    proposal_results: List[str] = []
    for out in proposal_outputs:
        text = out.outputs[0].text if out.outputs else ""
        proposal_results.append(text.strip())

    return idea_results, proposal_results


def detect_gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        try:
            count = len([d for d in cuda_visible.split(",") if d.strip() != ""])
            return max(1, count)
        except Exception:
            pass
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 'idea abstracts' and proposals for backtest papers using vLLM and Llama 3.1 8B Instruct."
        )
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name or path. Default: meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    tps = TENSOR_PARALLEL_SIZE_OVERRIDE if TENSOR_PARALLEL_SIZE_OVERRIDE is not None else detect_gpu_count()
    if tps is None or tps < 1:
        raise ValueError("Number of GPUs must be >= 1. Set TENSOR_PARALLEL_SIZE_OVERRIDE or ensure CUDA is visible.")

    # Hardcoded backtest input/output
    repo_root = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(os.path.dirname(__file__), "retrieved_papers", "backtest", "papers.json")
    ranks_path = os.path.join(os.path.dirname(__file__), "retrieved_papers", "backtest", "ranks_codes.json")
    output_dir = os.path.join(repo_root, "oracl", "data", "backtest")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "idea_abstracts_llama.json")

    records = load_records(input_path, ranks_path)
    
    if len(records) != 20:
        raise ValueError(f"Expected exactly 20 backtest papers, but found {len(records)}. Check papers.json and ranks_codes.json match.")

    idea_generations, proposal_generations = generate_idea_abstracts(
        records=records,
        model_name=args.model,
        tensor_parallel_size=tps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_records: List[Dict[str, Any]] = []
    for rec, idea, proposal in zip(records, idea_generations, proposal_generations):
        merged = dict(rec)
        merged["idea_abstract"] = idea
        merged["proposal"] = proposal
        output_records.append(merged)

    save_records(output_records, out_path)
    print(f"Wrote {len(output_records)} records to {out_path}")


if __name__ == "__main__":
    main()
