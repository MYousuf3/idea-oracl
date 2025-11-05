import os
import json
import tarfile
import re
from tqdm import tqdm
import tempfile
import shutil

def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def extract_abstract_from_latex(latex_content):
    """
    Extract abstract from LaTeX content using various patterns.
    Returns the abstract text or None if not found.
    """
    # Pattern 1: \begin{abstract}...\end{abstract}
    pattern1 = r'\\begin\{abstract\}(.*?)\\end\{abstract\}'
    match = re.search(pattern1, latex_content, re.DOTALL | re.IGNORECASE)
    if match:
        abstract = match.group(1).strip()
        return clean_latex_text(abstract)
    
    # Pattern 2: \abstract{...}
    pattern2 = r'\\abstract\{(.*?)\}'
    match = re.search(pattern2, latex_content, re.DOTALL | re.IGNORECASE)
    if match:
        abstract = match.group(1).strip()
        return clean_latex_text(abstract)
    
    # Pattern 3: Looking for abstract section with different delimiters
    pattern3 = r'\\begin\{abstract\*\}(.*?)\\end\{abstract\*\}'
    match = re.search(pattern3, latex_content, re.DOTALL | re.IGNORECASE)
    if match:
        abstract = match.group(1).strip()
        return clean_latex_text(abstract)
    
    return None

def clean_latex_text(text):
    """
    Clean LaTeX commands and formatting from text.
    """
    # Remove comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    
    # Remove common LaTeX commands but keep their content
    text = re.sub(r'\\textbf\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\textit\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\emph\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\cite\{.*?\}', '', text)
    text = re.sub(r'\\ref\{.*?\}', '', text)
    text = re.sub(r'\\label\{.*?\}', '', text)
    
    # Remove math mode delimiters but keep content
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    
    # Remove newlines and extra spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove remaining backslash commands
    text = re.sub(r'\\[a-zA-Z]+\{?\}?', '', text)
    text = re.sub(r'\\[^a-zA-Z]', '', text)
    
    # Clean up brackets
    text = re.sub(r'\{', '', text)
    text = re.sub(r'\}', '', text)
    
    return text.strip()

def find_main_tex_file(extracted_path):
    """
    Find the main .tex file in the extracted directory.
    Prioritizes files named main.tex, paper.tex, or files with \documentclass.
    """
    tex_files = []
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
    
    if not tex_files:
        return None
    
    # Check for common main file names
    for priority_name in ['main.tex', 'paper.tex', 'manuscript.tex', 'article.tex']:
        for tex_file in tex_files:
            if os.path.basename(tex_file).lower() == priority_name:
                return tex_file
    
    # Look for file with \documentclass (likely the main file)
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1000 chars
                if r'\documentclass' in content:
                    return tex_file
        except Exception:
            continue
    
    # If no main file found, return the first .tex file
    return tex_files[0] if tex_files else None

def extract_abstract_from_tar(tar_path):
    """
    Extract abstract from a tar.gz file containing LaTeX source.
    """
    try:
        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract tar file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
            
            # Find main .tex file
            main_tex = find_main_tex_file(temp_dir)
            if not main_tex:
                return None
            
            # Read the main tex file
            try:
                with open(main_tex, 'r', encoding='utf-8', errors='ignore') as f:
                    latex_content = f.read()
            except Exception as e:
                print(f"Error reading {main_tex}: {e}")
                return None
            
            # Extract abstract and return cleaned full content as well
            abstract = extract_abstract_from_latex(latex_content)
            cleaned_content = clean_full_latex(latex_content)
            return abstract, cleaned_content
            
    except Exception as e:
        print(f"Error extracting from {tar_path}: {e}")
        return None, None


def clean_full_latex(latex_content: str) -> str:
    """
    Produce a cleaned, plain-text version of the LaTeX source suitable for
    storing as `content`. This removes floats, listings, bibliographies,
    math, figures, and most LaTeX commands while preserving readable text.
    """
    if not latex_content:
        return ""

    s = latex_content

    # Remove comments
    s = re.sub(r'%.*$', '', s, flags=re.MULTILINE)

    # Remove common float/listing environments that are non-content
    remove_envs = [
        'figure', 'figure\*', 'table', 'table\*', 'lstlisting', 'algorithm',
        'align', 'align\*', 'equation', 'equation\*', 'thebibliography',
        'tikzpicture', 'tabular', 'verbatim', 'sidewaysfigure', 'sidebar'
    ]
    for env in remove_envs:
        s = re.sub(rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}', '', s, flags=re.DOTALL | re.IGNORECASE)

    # Remove includegraphics and other file includes
    s = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', '', s)
    s = re.sub(r'\\includegraphics\{.*?\}', '', s)
    s = re.sub(r'\\(input|include)\{.*?\}', '', s)

    # Remove bibliography commands
    s = re.sub(r'\\bibliography\{.*?\}', '', s)
    s = re.sub(r'\\bibliographystyle\{.*?\}', '', s)
    s = re.sub(r'\\printbibliography', '', s)

    # Remove citation commands
    s = re.sub(r'\\cite[t|p|alp|author|year]?\*?(?:\[.*?\])?\{.*?\}', '', s)

    # Remove labels, refs, and footnotes
    s = re.sub(r'\\label\{.*?\}', '', s)
    s = re.sub(r'\\ref\{.*?\}', '', s)
    s = re.sub(r'\\footnote\{.*?\}', '', s)

    # Remove displayed and inline math
    s = re.sub(r'\$\$(.*?)\$\$', '', s, flags=re.DOTALL)
    s = re.sub(r'\$(.*?)\$', '', s, flags=re.DOTALL)
    s = re.sub(r'\\\[(.*?)\\\]', '', s, flags=re.DOTALL)
    s = re.sub(r'\\\((.*?)\\\)', '', s, flags=re.DOTALL)

    # Remove begin/end markers left over
    s = re.sub(r'\\begin\{.*?\}', '', s)
    s = re.sub(r'\\end\{.*?\}', '', s)

    # Replace common sectioning commands with their title text
    s = re.sub(r'\\(?:section|subsection|subsubsection|paragraph|chapter)\*?\{(.*?)\}', r'\1', s)

    # Keep formatting commands' content (textbf, emph, etc.) and remove others
    s = clean_latex_text(s)

    # Collapse multiple newlines and whitespace
    s = re.sub(r'\r\n|\r', '\n', s)
    s = re.sub(r'\n\s*\n+', '\n\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = s.strip()

    return s

def process_conference_year(base_path, conference, year):
    """
    Process all papers for a specific conference and year.
    """
    path = os.path.join(base_path, conference.lower(), str(year))
    papers_file = os.path.join(path, 'papers.json')
    tar_path = os.path.join(path, 'tar_files')
    
    # Check if papers.json exists
    if not os.path.exists(papers_file):
        print(f"No papers.json found for {conference} {year}")
        return
    
    # Load papers data
    papers_data = open_json(papers_file)
    
    if not papers_data:
        print(f"No papers found in {conference} {year}")
        return
    
    print(f"\nProcessing {len(papers_data)} papers from {conference} {year}")
    
    updated_count = 0
    failed_count = 0
    
    for paper in tqdm(papers_data, desc=f"Extracting abstracts for {conference} {year}"):
        # Skip processing if both abstract and content already exist
        if ('abstract' in paper and paper['abstract']) and ('content' in paper and paper['content']):
            continue
        
        paper_id = paper.get('paper_id')
        if not paper_id:
            print(f"No paper_id found for submission {paper.get('submission_id')}")
            failed_count += 1
            continue
        
        # Path to tar file
        tar_file_path = os.path.join(tar_path, f"{paper_id}.tar.gz")
        
        if not os.path.exists(tar_file_path):
            print(f"Tar file not found: {tar_file_path}")
            failed_count += 1
            continue
        
        # Extract abstract and full content
        abstract, content = extract_abstract_from_tar(tar_file_path)

        if content:
            paper['content'] = content
        else:
            paper['content'] = ""

        if abstract:
            paper['abstract'] = abstract
            updated_count += 1
        else:
            # preserve existing abstract if present, otherwise set empty string
            if not paper.get('abstract'):
                paper['abstract'] = ""
            print(f"Could not extract abstract for {paper.get('title', 'Unknown')}")
            failed_count += 1
    
    # Save updated papers.json
    save_json(papers_data, papers_file)
    print(f"Updated {updated_count} papers, failed to extract {failed_count} abstracts")

def main():
    # Base path for all data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'retrieved_papers')
    
    if not os.path.exists(base_path):
        print(f"Retrieved papers directory not found: {base_path}")
        return
    
    # Conferences and years to process
    conferences = ['NeurIPS', 'ICLR', 'ICML']
    years = [2023, 2024, 2025]
    
    # Process each conference and year
    for conference in conferences:
        for year in years:
            conference_path = os.path.join(base_path, conference.lower(), str(year))
            if os.path.exists(os.path.join(conference_path, 'papers.json')):
                process_conference_year(base_path, conference, year)
            else:
                print(f"Skipping {conference} {year} - no papers.json found")
    
    print("\n✅ Abstract extraction complete!")

if __name__ == '__main__':
    main()

