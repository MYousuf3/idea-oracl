#!/usr/bin/env python3
"""
Open-source paper ranking using Llama 3 8B via vLLM.
Alternative to oracl_ranker.py (which uses OpenAI GPT models).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from .criteria_config import DEFAULT_CRITERIA, SYSTEM_PROMPT
except ImportError:
    from criteria_config import DEFAULT_CRITERIA, SYSTEM_PROMPT


# ===== User-editable GPU override =====
# Set to an integer >= 1 to force number of GPUs (tensor parallel size).
# Leave as None to auto-detect based on CUDA_VISIBLE_DEVICES or torch.cuda.
TENSOR_PARALLEL_SIZE_OVERRIDE = 8


@dataclass
class Paper:
    id: str
    title: str
    idea: str
    year: Optional[int] = None


def _build_llama_prompt(papers: Sequence[Paper], criteria: Sequence[str]) -> str:
    """Build a Llama 3 prompt for paper ranking using only idea abstracts."""
    joined_criteria = "\n".join(f"- {c}" for c in criteria)
    
    items = []
    for idx, p in enumerate(papers, start=1):
        year_str = f"\n   year={p.year}" if p.year is not None else ""
        items.append(
            f"{idx}. id={p.id}\n"
            f"   idea={p.idea}"
            f"{year_str}"
        )
    
    user_message = (
        "Produce a comparison-based total ranking of the papers below using the criteria.\n"
        "Focus on pairwise tradeoffs; avoid numeric scores.\n"
        "Rank the papers based solely on their idea abstracts (no titles provided).\n"
        "For each ranked item include only: position (1-based), id, concise rationale (<= 2 sentences).\n"
        "Return strictly valid JSON matching this schema:\n"
        '{"ranking": [{"position": int, "id": str, "rationale": str}, ...]}\n\n'
        f"Criteria:\n{joined_criteria}\n\n"
        f"Papers:\n{os.linesep.join(items)}\n\n"
        "Output only the JSON ranking, without any preamble or explanation."
    )
    
    # Llama 3.1 Instruct format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def rank_papers(
    papers: Sequence[Paper],
    criteria: Optional[Sequence[str]] = None,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    tensor_parallel_size: int = 1,
    max_new_tokens: int = 4096,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> List[Dict[str, Any]]:
    """Rank papers using Llama 3 8B via vLLM."""
    if not papers:
        return []

    if criteria is None or len(criteria) == 0:
        criteria = DEFAULT_CRITERIA

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

    prompt = _build_llama_prompt(papers, criteria)
    
    # Generate ranking
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    
    # Extract the response content
    try:
        output_text = outputs[0].outputs[0].text if outputs[0].outputs else ""
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(f"Unexpected vLLM response shape: {outputs}") from exc

    # Try to extract JSON from the output
    output_text = output_text.strip()
    
    # Sometimes models wrap JSON in markdown code blocks
    if output_text.startswith("```json"):
        output_text = output_text[7:]
    if output_text.startswith("```"):
        output_text = output_text[3:]
    if output_text.endswith("```"):
        output_text = output_text[:-3]
    output_text = output_text.strip()

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {output_text}") from exc

    ranking = data.get("ranking", [])

    # Ensure positions are contiguous starting at 1; if not present, infer from order
    if not ranking:
        return []

    # Normalize values and coerce expected keys
    normalized = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        pos = item.get("position")
        pid = item.get("id")
        title = item.get("title", "")  # Title not required from model output
        rationale = item.get("rationale")
        if pid is None:  # Only require ID from model output
            continue
        try:
            pos_int = int(pos) if pos is not None else None
        except Exception:  # noqa: BLE001
            pos_int = None
        normalized.append({
            "position": pos_int,
            "id": str(pid),
            "title": str(title),  # Will be empty if not provided, filled later from original data
            "rationale": str(rationale) if rationale is not None else "",
        })

    # If any position missing, assign based on current order
    if any(n["position"] is None for n in normalized):
        for i, n in enumerate(normalized, start=1):
            n["position"] = i

    # Sort by position asc, then reassign to ensure 1..N without gaps
    normalized.sort(key=lambda x: int(x.get("position", 10**9)))
    for i, n in enumerate(normalized, start=1):
        n["position"] = i

    # Ensure every provided id is present; append missing in original order
    provided_ids_order = [p.id for p in papers]
    present_ids = {n["id"] for n in normalized}
    for pid in provided_ids_order:
        if pid not in present_ids:
            # Find title from input papers
            p = next((pp for pp in papers if pp.id == pid), None)
            normalized.append({
                "position": len(normalized) + 1,
                "id": pid,
                "title": p.title if p else "",
                "rationale": "Appended to complete total order (model omitted item).",
            })

    # Reassign positions one final time
    normalized.sort(key=lambda x: int(x.get("position", 10**9)))
    for i, n in enumerate(normalized, start=1):
        n["position"] = i

    return normalized


def rank_papers_from_dicts(
    papers: Sequence[Dict[str, Any]],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Rank papers from dictionary format and merge with original data."""
    # Create Paper objects using idea_abstract if available, otherwise idea
    as_objs = [
        Paper(
            id=str(p.get("id", p.get("submission_id", p.get("paper_id", "")))),
            title=str(p.get("title", "")),
            idea=str(p.get("idea_abstract", p.get("idea", ""))),
            year=int(p["year"]) if "year" in p and p["year"] is not None else None,
        )
        for p in papers
    ]
    
    # Get ranking from Llama
    ranking_results = rank_papers(as_objs, **kwargs)
    
    # Create a map of original papers by their IDs for quick lookup
    id_to_paper = {}
    for p in papers:
        pid = str(p.get("id", p.get("submission_id", p.get("paper_id", ""))))
        id_to_paper[pid] = p
    
    # Merge ranking results with original paper data
    enriched_results = []
    for ranked_item in ranking_results:
        pid = ranked_item["id"]
        original_paper = id_to_paper.get(pid, {})
        
        # Start with all original fields
        merged_item = dict(original_paper)
        
        # Add ranking-specific fields
        merged_item["position"] = ranked_item["position"]
        merged_item["rationale"] = ranked_item["rationale"]
        
        enriched_results.append(merged_item)
    
    return enriched_results


def compute_default_output_path(input_path: str) -> str:
    """
    Compute output path for rankings.
    Maps .../finished_data/<conf>/<year>/idea_abstracts*.json 
    -> .../oracl/data/<conf>/<year>/ranked_papers_llama.json
    """
    norm = os.path.normpath(input_path)
    parts = norm.split(os.sep)
    
    # Extract conference and year from path
    try:
        idx = parts.index("finished_data")
        base_parts = parts[:idx]  # Everything before finished_data
        conf = parts[idx + 1] if len(parts) > idx + 1 else "unknown_conf"
        year = parts[idx + 2] if len(parts) > idx + 2 else "unknown_year"
        
        # Find oracl directory in base path
        if "oracl" in base_parts:
            oracl_idx = base_parts.index("oracl")
            base = os.sep.join(parts[:oracl_idx + 1])
        else:
            # Default to parent of finished_data
            base = os.sep.join(base_parts)
            base = os.path.join(base, "oracl")
        
        return os.path.join(base, "data", conf, year, "ranked_papers_llama.json")
    except (ValueError, IndexError):
        # Fallback: save to oracl/data/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "data", "ranked_papers_llama.json")


def detect_gpu_count() -> int:
    """Detect number of available GPUs from CUDA_VISIBLE_DEVICES."""
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
        description="Rank papers using Llama 3 8B via vLLM. Input: JSON with papers. "
                   "Output: Ranked JSON with all original fields + position and rationale."
    )
    parser.add_argument("input_json", help="Path to input JSON list of papers")
    parser.add_argument(
        "-o", "--output",
        help="Output path for ranked JSON. If not provided, saves to oracl/data/<conf>/<year>/ranked_papers_llama.json"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B",
        help="HF model name or path. Default: meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        help="Custom ranking criteria (space-separated). If not provided, uses default criteria."
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()

    # Use in-code override or auto-detect
    tps = TENSOR_PARALLEL_SIZE_OVERRIDE if TENSOR_PARALLEL_SIZE_OVERRIDE is not None else detect_gpu_count()
    if tps is None or tps < 1:
        raise ValueError("Number of GPUs must be >= 1. Set TENSOR_PARALLEL_SIZE_OVERRIDE or ensure CUDA is visible.")

    with open(args.input_json, "r", encoding="utf-8") as f:
        papers_dicts = json.load(f)

    kwargs = {
        "model_name": args.model,
        "tensor_parallel_size": tps,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.criteria:
        kwargs["criteria"] = args.criteria

    print(f"Ranking {len(papers_dicts)} papers using {args.model} with {tps} GPU(s)...")
    results = rank_papers_from_dicts(papers_dicts, **kwargs)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = compute_default_output_path(args.input_json)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Ranked {len(results)} papers")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

