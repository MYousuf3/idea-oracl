"""
Shared ranking criteria configuration for paper evaluation.
Used by both OpenAI-based and Llama-based rankers.
"""

DEFAULT_CRITERIA = (
    "long-term scientific impact",
    "real-world adoption and influence",
    "methodological soundness and novelty",
    "community consensus and reproducibility",
)

SYSTEM_PROMPT = (
    "You are a rigorous research evaluator. Score fairly, avoid popularity bias,"
    " and use present-day knowledge of real-world impact. Derive a clear total order"
    " using direct comparisons rather than numeric scoring."
)

