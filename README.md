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
