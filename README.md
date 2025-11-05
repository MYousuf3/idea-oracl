Idea Optimized Research Assessment and Concept LLM (Idea ORACL)

Assessing and scoring academic papers based on ideas, to save time on scientific discovery.

## Setup

### 1. Create and Activate Conda Environment

First, create the conda environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate idea-oracl
```

## Data Collection

The `data-collection/` directory contains scripts to collect papers from major ML conferences (NeurIPS, ICLR, ICML) and extract their abstracts.

### 2. Collecting Papers

The `collect_papers.py` script downloads papers from OpenReview and arXiv.

**Configure number of papers per conference:**

Edit `data-collection/collect_papers.py` and modify the `max_papers` variable in the `main()` function:

```python
max_papers = 3  # Change this to download more papers per conference
```

**Run the collection script:**

```bash
cd data-collection
python collect_papers.py
```

This will:
- Download papers from NeurIPS, ICLR, and ICML (2023-2024)
- Save paper metadata to `retrieved_papers/{conference}/{year}/papers.json`
- Download LaTeX source files to `retrieved_papers/{conference}/{year}/tar_files/`
- Extract source files to `retrieved_papers/{conference}/{year}/extracted_files/`

The script automatically:
- Tries both OpenReview API v1 and v2
- Skips conferences/years that already have the requested number of papers
- Respects rate limits with sleep timers

### 3. Extracting Abstracts

After collecting papers, run the abstract extraction script:

```bash
python extract_abstracts.py
```

This will:
- Read all `papers.json` files across conferences
- Extract abstracts from LaTeX source files
- Add an `abstract` field to each paper in the JSON
- Clean LaTeX formatting for readable text

The script is idempotent - it skips papers that already have abstracts, so you can safely run it multiple times.

## Project Structure

```
idea-oracl/
├── data-collection/
│   ├── collect_papers.py          # Download papers from conferences
│   ├── extract_abstracts.py       # Extract abstracts from LaTeX sources
│   └── retrieved_papers/          # Downloaded papers organized by conference/year
│       ├── neurips/
│       ├── iclr/
│       └── icml/
├── oracl/                          # Main ORACL modules
├── environment.yml                 # Conda environment specification
└── README.md
```

## Pipeline: proposals, ranking and metrics

The following steps extend the collection/extraction pipeline to generate compact proposals for each paper, run the ranker using those proposals, and compute simple metrics comparing the new ordering to the original `rank` field.

4. Generate proposals

- Script: `data-collection/idea_extrapolation_llama.py` (and `..._backtest.py` for the backtest set)
- What it does: reads extracted papers (with `abstract` and optionally cleaned `content`) and generates a short `proposal` field for each paper. The result is written as a JSON list (for backtest: `oracl/data/backtest/idea_abstracts_llama.json`).
- Why: proposals are compact, focused prompts the ranker uses instead of the full paper content to improve stability and avoid large prompt sizes.

Example:

```bash
python3 data-collection/idea_extrapolation_llama_backtest.py
```

5. Rank using proposals

- Script: `oracl/oracl_ranker.py`
- What it does: loads a JSON list of papers (expects `title`, `idea_abstract`, ideally `proposal`), builds a prompt using the `proposal` field (deliberately excluding `content`), calls the LLM using a strict JSON schema, and post-processes the returned ranking.
- Important behaviors:
	- If an input `id` is missing the ranker computes a stable fallback id (prefers `id` → `arxiv` or `paper_tar` → `paper_<index>`).
	- Model-returned ids that do not match any input id are ignored (this filters hallucinated items like invented ids).
	- Duplicate ids from the model are deduplicated; omitted input papers are appended so the final output is a full total order.
	- The ranker no longer depends on `retrieved_papers/backtest/ranks_codes.json` — it ranks purely on the supplied input list.

Example (backtest):

```bash
cd oracl
python3 oracl_ranker.py
```

Output: `oracl/data/backtest/ranked_papers.json`

6. Compute ranking metrics

- Script: `tools/compute_rank_metrics.py`
- What it does: compares each paper's original `rank` to the ranker's `position` and prints summary statistics and top movers.
- Metrics include: mean signed difference, mean absolute difference, mean squared difference (MSE), median absolute difference, max absolute difference, number unchanged, and top movers by absolute change.

Usage:

```bash
python3 tools/compute_rank_metrics.py oracl/data/backtest/ranked_papers.json
```

The script is dependency-free and prints a human-readable summary to stdout. If you want CSV/JSON export, RMSE, or correlation statistics, we can add flags for those.

