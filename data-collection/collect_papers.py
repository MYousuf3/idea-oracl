import openreview
import os
import subprocess
import arxiv
import json
from tqdm import tqdm
import time
import requests

def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def get_arxiv_id(client, title):
    try:
        search = arxiv.Search(
            query=f"ti:{title}",
            max_results=1
        )
        
        result = next(client.results(search), None)
        if result is not None:
            url = str(result)
            return url.split('/')[-1]
        
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Could not find paper_id for {title}")
        return None

def get_citation_count(arxiv_id, title):
    """
    Get citation count from Semantic Scholar API using arXiv ID or title.
    Returns the citation count or None if not found.
    """
    try:
        # First try with arXiv ID (more reliable)
        if arxiv_id:
            url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
            params = {'fields': 'citationCount'}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                citation_count = data.get('citationCount', 0)
                return citation_count if citation_count is not None else 0
            elif response.status_code == 404:
                print(f"Paper not found in Semantic Scholar: {arxiv_id}")
            else:
                print(f"Semantic Scholar API error (status {response.status_code}) for {arxiv_id}")
        
        # Fallback: try searching by title if arXiv ID didn't work
        if title:
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': title,
                'fields': 'citationCount,title',
                'limit': 1
            }
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    citation_count = data['data'][0].get('citationCount', 0)
                    return citation_count if citation_count is not None else 0
        
        return None
    except Exception as e:
        print(f"Error fetching citation count: {e}")
        return None

def download_paper(client, paper_id, path):
    try:
        paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        paper.download_source(filename=os.path.join(path, f"tar_files/{paper_id}.tar.gz"))
        return True
    except Exception as e:
        print(f"Error downloading paper {paper_id}: {e}")
        return False

def collect_papers(client, openreview_clients, conference, year, base_path, max_papers=3):
    path = os.path.join(base_path, conference.lower(), str(year))
    os.makedirs(os.path.join(path, 'tar_files'), exist_ok=True)
    os.makedirs(os.path.join(path, 'extracted_files'), exist_ok=True)

    # Initialize or load existing papers data
    papers_file = os.path.join(path, 'papers_with_citations.json')
    if os.path.exists(papers_file):
        papers_data = open_json(papers_file)
        existing_ids = set(paper['submission_id'] for paper in papers_data)
        # Start counter from existing papers count
        papers_collected = len(papers_data)
    else:
        papers_data = []
        existing_ids = set()
        papers_collected = 0

    # Check if we already have enough papers
    if papers_collected >= max_papers:
        print(f"Already have {papers_collected} papers for {conference} {year}, skipping")
        return

    # Try different API clients and invitation formats
    # API v1 and v2 have different formats
    api_configs = [
        {
            'client': openreview_clients['v2'],
            'invitations': [
                f"{conference}.cc/{year}/Conference/-/Submission",
                f"{conference}.cc/{year}/Conference/-/Blind_Submission",
            ],
            'api_version': 'v2'
        },
        {
            'client': openreview_clients['v1'],
            'invitations': [
                f"{conference}.cc/{year}/Conference/-/Blind_Submission",
                f"{conference}.cc/{year}/Conference/-/Submission",
            ],
            'api_version': 'v1'
        }
    ]
    
    submissions = None
    successful_api = None
    
    for api_config in api_configs:
        for invitation in api_config['invitations']:
            try:
                print(f"Trying {api_config['api_version']} with invitation: {invitation}")
                submissions = api_config['client'].get_all_notes(invitation=invitation)
                if submissions:
                    print(f"Retrieved {len(submissions)} submissions from {conference} {year} using {api_config['api_version']} API with {invitation}")
                    successful_api = api_config['api_version']
                    break
            except Exception as e:
                print(f"Failed with {api_config['api_version']} API and invitation '{invitation}': {str(e)[:100]}")
                continue
        if submissions:
            break
    
    if not submissions:
        print(f"Could not retrieve submissions for {conference} {year} with any known API version or invitation format")
        return

    previous_title = None
    previous_paper_id = None

    for submission in tqdm(submissions, desc=f"Processing {conference} {year}"):
        if papers_collected >= max_papers:
            print(f"Reached maximum papers ({max_papers}) for {conference} {year}")
            break

        if submission.id in existing_ids:
            continue

        try:
            # Handle both API v1 and v2 content formats
            if hasattr(submission, 'content'):
                if isinstance(submission.content, dict):
                    title = submission.content.get('title', {})
                    # In API v2, content fields might be dicts with 'value' key
                    if isinstance(title, dict):
                        title = title.get('value', '')
                else:
                    title = str(submission.content)
            else:
                print(f"Submission {submission.id} has no content, skipping")
                continue
            
            if not title:
                print(f"Submission {submission.id} has no title, skipping")
                continue
            
            # Reuse arxiv ID if same title (optimization)
            if title == previous_title:
                paper_id = previous_paper_id
            else:
                print(f"Collecting paper_id for {title}")
                paper_id = get_arxiv_id(client, title)
                previous_paper_id = paper_id
                previous_title = title

            # Sleep to respect rate limits
            time.sleep(1)
            
            if paper_id:
                paper_path = os.path.join(path, f"tar_files/{paper_id}.tar.gz")
                
                # Get citation count from Semantic Scholar
                print(f"Fetching citation count for {paper_id}")
                citation_count = get_citation_count(paper_id, title)
                if citation_count is not None:
                    print(f"Found {citation_count} citations")
                else:
                    print(f"Could not fetch citation count")
                
                # Respect Semantic Scholar rate limits (100 req/5min = ~1 req/3sec)
                time.sleep(3)
                
                if not os.path.exists(paper_path):
                    print(f"Downloading paper {paper_id}")
                    if download_paper(client, paper_id, path):
                        paper_info = {
                            'submission_id': submission.id,
                            'paper_id': paper_id,
                            'title': title,
                            'conference': conference,
                            'year': year,
                            'citation_count': citation_count
                        }
                        papers_data.append(paper_info)
                        papers_collected += 1
                        save_json(papers_data, papers_file)
                        print(f"Collected {papers_collected}/{max_papers} papers for {conference} {year}")
                else:
                    # Paper tar exists but not in JSON - add to JSON and count it
                    print(f"Found existing paper {paper_id}")
                    paper_info = {
                        'submission_id': submission.id,
                        'paper_id': paper_id,
                        'title': title,
                        'conference': conference,
                        'year': year,
                        'citation_count': citation_count
                    }
                    papers_data.append(paper_info)
                    papers_collected += 1
                    save_json(papers_data, papers_file)
                    print(f"Added existing paper to JSON, {papers_collected}/{max_papers} papers for {conference} {year}")

        except Exception as e:
            print(f"Error processing submission {submission.id}: {e}")
            continue

def unpack_tar_files(base_path, conference, year):
    path = os.path.join(base_path, conference.lower(), str(year))
    tar_path = os.path.join(path, 'tar_files')
    extract_path = os.path.join(path, 'extracted_files')
    
    if not os.path.exists(tar_path):
        print(f"No tar files directory found for {conference} {year}")
        return

    for file in tqdm(os.listdir(tar_path), desc=f"Extracting {conference} {year}"):
        file_path = os.path.join(tar_path, file)
        paper_extract_path = os.path.join(extract_path, os.path.splitext(file)[0])
        os.makedirs(paper_extract_path, exist_ok=True)
        
        if file.endswith('.tar.gz') or file.endswith('.tgz'):
            command = ['tar', '-xzf', file_path, '-C', paper_extract_path]
        elif file.endswith('.tar'):
            command = ['tar', '-xf', file_path, '-C', paper_extract_path]
        else:
            print(f"Unsupported file format: {file}")
            continue
        
        try:
            result = subprocess.run(command, capture_output=True)
            if result.returncode != 0:
                print(f"Error extracting {file}: {result.stderr.decode('utf-8')}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

def main():
    # Initialize clients
    arxiv_client = arxiv.Client()
    
    # Initialize both OpenReview API v1 and v2 clients
    openreview_clients = {
        'v1': openreview.Client(baseurl='https://api.openreview.net'),
        'v2': openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    }
    
    print("Initialized OpenReview API v1 and v2 clients")
    
    # Base path for all data (now inside data-collection directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'retrieved_papers')
    os.makedirs(base_path, exist_ok=True)
    
    # Conferences and years to process
    conferences = ['NeurIPS', 'ICLR', 'ICML']
    years = [2023, 2024]  # 2025 may not have papers available yet
    max_papers = 15 # change this to download more than 3 papers 
    # Process each conference and year
    for conference in conferences:
        for year in years:
            print(f"\nProcessing {conference} {year}")
            collect_papers(arxiv_client, openreview_clients, conference, year, base_path, max_papers)
            unpack_tar_files(base_path, conference, year)

if __name__ == '__main__':
    main()