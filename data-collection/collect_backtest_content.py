#!/usr/bin/env python3
"""
Collect and extract content for papers listed in retrieved_papers/backtest/ranks_codes.json.
This mirrors the style of collect_papers.py but works from a local list of arXiv IDs.

For each entry, it will:
- download the paper source tarball to backtest/tar_files/<id>.tar.gz (if missing)
- extract the main .tex and produce cleaned `content` and `abstract` using
  functions from extract_abstracts.py
- write (or update) backtest/papers.json with collected metadata and content

"""
import os
import json
import time
from tqdm import tqdm
import arxiv

from extract_abstracts import extract_abstract_from_tar, save_json, open_json


def ensure_dirs(base_path: str):
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'tar_files'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'extracted_files'), exist_ok=True)


def load_ranks(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def download_source(client: arxiv.Client, paper_id: str, dest_path: str) -> bool:
    try:
        search = arxiv.Search(id_list=[paper_id])
        result = next(client.results(search), None)
        if result is None:
            print(f"No arXiv result for {paper_id}")
            return False
        result.download_source(filename=dest_path)
        return True
    except Exception as e:
        print(f"Error downloading {paper_id}: {e}")
        return False


def collect_backtest(base_dir: str):
    ranks_path = os.path.join(base_dir, 'ranks_codes.json')
    if not os.path.exists(ranks_path):
        raise FileNotFoundError(f"ranks_codes.json not found at {ranks_path}")

    ranks = load_ranks(ranks_path)

    backtest_path = base_dir
    ensure_dirs(backtest_path)

    papers_file = os.path.join(backtest_path, 'papers.json')
    if os.path.exists(papers_file):
        papers_data = open_json(papers_file)
        existing_ids = set(p['arxiv'] for p in papers_data if 'arxiv' in p)
    else:
        papers_data = []
        existing_ids = set()

    arxiv_client = arxiv.Client()

    for entry in tqdm(ranks, desc='Collecting backtest papers'):
        paper_id = entry.get('arxiv') or entry.get('paper_id')
        title = entry.get('title')
        rank = entry.get('rank')

        if not paper_id:
            print(f"Skipping entry without arXiv id: {entry}")
            continue

        # Normalize filename (use the arXiv id as-is)
        tar_filename = f"{paper_id}.tar.gz"
        tar_path = os.path.join(backtest_path, 'tar_files', tar_filename)

        if paper_id in existing_ids:
            # already recorded; skip downloading but ensure content exists
            # find existing record
            # we will still attempt to extract/refresh content if missing
            recs = [p for p in papers_data if p.get('arxiv') == paper_id]
            existing = recs[0] if recs else None
        else:
            existing = None

        if not os.path.exists(tar_path):
            print(f"Downloading source for {paper_id} ({title})")
            ok = download_source(arxiv_client, paper_id, tar_path)
            # be polite with arXiv
            time.sleep(1)
            if not ok:
                print(f"Failed to download {paper_id}")
                continue

        # Extract abstract and cleaned content from tar
        abstract, content = extract_abstract_from_tar(tar_path)

        record = {
            'rank': rank,
            'title': title,
            'arxiv': paper_id,
            'paper_tar': os.path.relpath(tar_path, start=base_dir),
            'abstract': abstract or "",
            'content': content or "",
        }

        if existing:
            # update existing entry
            existing.update(record)
        else:
            papers_data.append(record)
            existing_ids.add(paper_id)

        # Save after each paper to avoid losing progress
        save_json(papers_data, papers_file)

    print(f"Wrote {len(papers_data)} records to {papers_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'retrieved_papers', 'backtest')
    collect_backtest(base_path)


if __name__ == '__main__':
    main()
