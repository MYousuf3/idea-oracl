

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

try:
    from .criteria_config import DEFAULT_CRITERIA, SYSTEM_PROMPT
except ImportError:
    from criteria_config import DEFAULT_CRITERIA, SYSTEM_PROMPT


@dataclass
class Paper:
    id: str
    title: str
    idea: str
    year: Optional[int] = None


def _build_prompt(papers: Sequence[Paper], criteria: Sequence[str]) -> Dict[str, str]:
    joined_criteria = "\n".join(f"- {c}" for c in criteria)
    items = []
    for idx, p in enumerate(papers, start=1):
        year_str = f"\n   year={p.year}" if p.year is not None else ""
        items.append(
            f"{idx}. id={p.id}\n"
            f"   title={p.title}\n"
            f"   idea={p.idea}"
            f"{year_str}"
        )
    user = (
        "Produce a comparison-based total ranking of the papers below using the criteria.\n"
        "Focus on pairwise tradeoffs; avoid numeric scores.\n"
        "For each ranked item include only: position (1-based), id, title, concise rationale (<= 2 sentences).\n"
        "Return strictly the JSON schema.\n\n"
        f"Criteria:\n{joined_criteria}\n\n"
        f"Papers:\n{os.linesep.join(items)}"
    )
    return {"system": SYSTEM_PROMPT, "user": user}


def rank_papers(
    papers: Sequence[Paper],
    criteria: Optional[Sequence[str]] = None,
    model: str = "gpt-5",
    temperature: float = 1,
) -> List[Dict[str, Any]]:
    if not papers:
        return []

    if criteria is None or len(criteria) == 0:
        criteria = DEFAULT_CRITERIA

    client = OpenAI()

    schema = {
        "name": "paper_ranking",
        "schema": {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "integer"},
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["position", "id", "title", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ranking"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    prompts = _build_prompt(papers, criteria)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ],
        temperature=temperature,
        response_format={"type": "json_schema", "json_schema": schema},
    )

    # Extract the response content
    try:
        output_text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        raise RuntimeError(f"Unexpected response shape: {response}") from exc

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
        title = item.get("title")
        rationale = item.get("rationale")
        if pid is None or title is None:
            continue
        try:
            pos_int = int(pos) if pos is not None else None
        except Exception:  # noqa: BLE001
            pos_int = None
        normalized.append({
            "position": pos_int,
            "id": str(pid),
            "title": str(title),
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
    
    # Get ranking from GPT
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
    -> .../oracl/data/<conf>/<year>/ranked_papers.json
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
        
        return os.path.join(base, "data", conf, year, "ranked_papers.json")
    except (ValueError, IndexError):
        # Fallback: save to oracl/data/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "data", "ranked_papers.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Rank papers using GPT. Input: JSON with papers. Output: Ranked JSON with all original fields + position and rationale."
    )
    parser.add_argument("input_json", help="Path to input JSON list of papers")
    parser.add_argument(
        "-o", "--output",
        help="Output path for ranked JSON. If not provided, saves to oracl/data/<conf>/<year>/ranked_papers.json"
    )
    parser.add_argument("--model", default="gpt-5", help="OpenAI model to use (default: gpt-5)")
    parser.add_argument(
        "--criteria",
        nargs="+",
        help="Custom ranking criteria (space-separated). If not provided, uses default criteria."
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        papers_dicts = json.load(f)

    kwargs = {"model": args.model}
    if args.criteria:
        kwargs["criteria"] = args.criteria

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

