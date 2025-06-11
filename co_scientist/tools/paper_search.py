import requests
import xml.etree.ElementTree as ET
import time
import urllib.parse
import os
import json
import hashlib
from pdfminer.high_level import extract_text
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# arXiv base URL
BASE_URL = 'http://export.arxiv.org/api/query'

# Directories for saving results
PDF_DIR = "pdfs"
TXT_DIR = "txts"
META_DIR = "metadata"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# Set up session with retry strategy
session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Parse arXiv Atom feed
def parse_arxiv_response(xml_response):
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    root = ET.fromstring(xml_response)
    papers = []

    for entry in root.findall('atom:entry', ns):
        paper = {}
        paper['id'] = entry.find('atom:id', ns).text
        paper['title'] = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        paper['summary'] = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        paper['published'] = entry.find('atom:published', ns).text

        authors = entry.findall('atom:author/atom:name', ns)
        paper['authors'] = [author.text for author in authors]

        primary_cat = entry.find('arxiv:primary_category', ns)
        paper['primary_category'] = primary_cat.attrib['term'] if primary_cat is not None else None

        links = entry.findall('atom:link', ns)
        for link in links:
            if link.attrib.get('title') == 'pdf':
                paper['pdf_url'] = link.attrib['href']
                break

        papers.append(paper)

    return papers

# Keyword search
def keyword_search(query, start=0, max_results=10, sortBy="relevance", sortOrder="descending"):
    search_query = urllib.parse.quote(f"all:{query}")
    url = f"{BASE_URL}?search_query={search_query}&start={start}&max_results={max_results}&sortBy={sortBy}&sortOrder={sortOrder}"

    response = session.get(url)
    if response.status_code == 200:
        return parse_arxiv_response(response.text)
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

# Download PDF file
def download_pdf(pdf_url, paper_id):
    pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
    if os.path.exists(pdf_path):
        print(f"PDF already exists: {pdf_path}")
        return pdf_path

    response = session.get(pdf_url)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded PDF: {pdf_path}")
    return pdf_path

# Convert PDF to TXT
def pdf_to_text(pdf_path, txt_path):
    if os.path.exists(txt_path):
        print(f"TXT already exists: {txt_path}")
        return

    text = extract_text(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved TXT: {txt_path}")

# Generate hash for deduplication
def hash_paper(paper):
    return hashlib.md5(paper['id'].encode()).hexdigest()

# Main function
if __name__ == '__main__':
    queries = ["spatial clustering", "urban geography", "GIS"]
    processed_hashes = set()

    for query in queries:
        print(f"\nSearching arXiv papers for: '{query}'")
        papers = keyword_search(query, max_results=20)

        print(f"Found {len(papers)} papers.\n")
        for paper in papers:
            paper_hash = hash_paper(paper)
            if paper_hash in processed_hashes:
                print(f"Duplicate found, skipping: {paper['title']}")
                continue

            arxiv_id = paper['id'].split('/')[-1]
            pdf_url = paper.get('pdf_url')
            pdf_path = download_pdf(pdf_url, arxiv_id)

            txt_path = os.path.join(TXT_DIR, f"{arxiv_id}.txt")
            pdf_to_text(pdf_path, txt_path)

            meta_path = os.path.join(META_DIR, f"{arxiv_id}.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2)

            processed_hashes.add(paper_hash)
            time.sleep(3)

    print("\nAll tasks completed.")
