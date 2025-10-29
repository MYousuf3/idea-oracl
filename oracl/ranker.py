

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI


@dataclass
class Paper:
    id: str
    title: str
    idea: str


def _build_prompt(papers: Sequence[Paper], criteria: Sequence[str]) -> Dict[str, str]:
    joined_criteria = "\n".join(f"- {c}" for c in criteria)
    items = []
    for idx, p in enumerate(papers, start=1):
        items.append(
            f"{idx}. id={p.id}\n"
            f"   title={p.title}\n"
            f"   idea={p.idea}"
        )
    user = (
        "Produce a comparison-based total ranking of the papers below using the criteria.\n"
        "Focus on pairwise tradeoffs; avoid numeric scores.\n"
        "For each ranked item include only: position (1-based), id, title, concise rationale (<= 2 sentences).\n"
        "Return strictly the JSON schema.\n\n"
        f"Criteria:\n{joined_criteria}\n\n"
        f"Papers:\n{os.linesep.join(items)}"
    )
    system = (
        "You are a rigorous research evaluator. Score fairly, avoid popularity bias,"
        " and use present-day knowledge of real-world impact. Derive a clear total order"
        " using direct comparisons rather than numeric scoring."
    )
    return {"system": system, "user": user}


def rank_papers(
    papers: Sequence[Paper],
    criteria: Optional[Sequence[str]] = None,
    model: str = "gpt-5",
    temperature: float = 0.2,
) -> List[Dict[str, Any]]:
    if not papers:
        return []

    if criteria is None or len(criteria) == 0:
        criteria = (
            "long-term scientific impact",
            "real-world adoption and influence",
            "methodological soundness and novelty",
            "community consensus and reproducibility",
        )

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

    response = client.responses.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ],
        temperature=temperature,
        response_format={"type": "json_schema", "json_schema": schema},
    )

    output_text = getattr(response, "output_text", None)
    if not output_text:
        # Fallback for older SDK shapes
        try:
            output_text = response.output[0].content[0].text
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected response shape: {response}") from exc

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as exc:  # noqa: F841
        raise ValueError("Model did not return valid JSON.")

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
    as_objs = [
        Paper(
            id=str(p["id"]),
            title=str(p.get("title", "")),
            idea=str(p["idea"]),
        )
        for p in papers
    ]
    return rank_papers(as_objs, **kwargs)


if __name__ == "__main__":
    # Minimal CLI example: read a JSON list of {id,title,idea}
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to input JSON list of papers")
    parser.add_argument("--model", default="gpt-5")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        papers_dicts = json.load(f)

    results = rank_papers_from_dicts(papers_dicts, model=args.model)
    print(json.dumps(results, ensure_ascii=False, indent=2))

