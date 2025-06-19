import requests
import os
from tqdm import tqdm
from pdfminer.high_level import extract_text
from prompt import prompt_manager
from engine.base_engine import LLMEngine
p_manager = prompt_manager.PromptManager("/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/prompt")

llm_engine = LLMEngine('gpt-4.1')
def SemanticScholarHighCiteSearch(refine_keywords:str, min_citation_count: int = 50,top_n:int = 50, year_from:int = 2015):
    """search high cited paper from Semantic Scholar API based on the user query"""
    query_params = {
        "query": refine_keywords,
        "fields": "title,authors,citationCount,url,year,abstract,openAccessPdf",
        "year": f"{year_from}-",
        "minCitationCount": str(min_citation_count)
    }
    response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params=query_params, headers={"x-api-key": "opjXWQ6vW46i5970FzSut2IuJhEWhXLB9Oh28GmK"})
    if response.status_code != 200:
        print(f"API error: {response.status_code} - {response.text}")
        return []
    result = response.json()
    papers = result.get("data", [])
    papers_sorted = sorted(papers, key=lambda x: x.get("citationCount", 0), reverse=True)[:top_n]

    paper_infos = []

    for paper in papers_sorted:
        info = {
            "title": paper.get("title"),
            "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
            "year": paper.get("year"),
            "citationCount": paper.get("citationCount"),
            "url": paper.get("url"),
            "abstract": paper.get("abstract", ""),
            "openAccessPdf": paper.get("openAccessPdf", {}).get("url")
        }

        paper_infos.append(info)

    papers_filtered = [p for p in paper_infos if p.get("abstract") is not None]

    return papers_filtered

class SemanticScholarHighCiteFetcher:
    def __init__(self, api_key, year_from=2020):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        self.headers = {"x-api-key": self.api_key}
        self.year_from = year_from  # 默认只查近几年

    def search_by_query(self, query, min_citation_count=10,top_n=50):
        query_params = {
            "query": query,
            "fields": "title,authors,citationCount,url,year,abstract,openAccessPdf",
            "year": f"{self.year_from}-",
            "minCitationCount": str(min_citation_count)
        }
        response = requests.get(self.base_url, params=query_params, headers=self.headers)
        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            return []
        result = response.json()
        papers = result.get("data", [])
        papers_sorted = sorted(papers, key=lambda x: x.get("citationCount", 0), reverse=True)[:top_n]

        paper_infos = []

        for paper in papers_sorted:
            info = {
                "title": paper.get("title"),
                "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount"),
                "url": paper.get("url"),
                "abstract": paper.get("abstract", ""),
                "openAccessPdf": paper.get("openAccessPdf", {}).get("url")
            }

            paper_infos.append(info)

        papers_filtered = [p for p in paper_infos if p.get("abstract") is not None]

        return papers_filtered

    # def download_pdf(self, paper, pdf_dir=None):
    #     if pdf_dir is None:
    #         pdf_dir = os.path.join(self.save_dir, "pdfs")
    #     os.makedirs(pdf_dir, exist_ok=True)
    #
    #     pdf_url = None
    #     if paper.get("openAccessPdf") and paper["openAccessPdf"].get("url"):
    #         pdf_url = paper["openAccessPdf"]["url"]
    #     if not pdf_url:
    #         return None  # 无开放PDF
    #     title = paper.get("title", "paper").replace("/", "_").replace(" ", "_")[:100]
    #     filename = f"{title}_{paper['year']}_{paper['paperId'][:6]}.pdf"
    #     file_path = os.path.join(pdf_dir, filename)
    #
    #     if os.path.exists(file_path):
    #         return file_path  # 已下载
    #
    #     try:
    #         with requests.get(pdf_url, stream=True, timeout=30) as r:
    #             r.raise_for_status()
    #             with open(file_path, "wb") as f:
    #                 for chunk in r.iter_content(chunk_size=8192):
    #                     if chunk:
    #                         f.write(chunk)
    #         return file_path
    #     except Exception as e:
    #         print(f"Download failed for {pdf_url}: {e}")
    #         return None
    #
    # def pdf_to_txt(self, pdf_path, txt_dir=None):
    #     if txt_dir is None:
    #         txt_dir = os.path.join(self.save_dir, "txts")
    #     os.makedirs(txt_dir, exist_ok=True)
    #     base = os.path.basename(pdf_path)
    #     txt_path = os.path.join(txt_dir, base.replace(".pdf", ".txt"))
    #     try:
    #         text = extract_text(pdf_path)
    #         with open(txt_path, "w", encoding="utf-8") as f:
    #             f.write(text)
    #         return txt_path
    #     except Exception as e:
    #         print(f"Failed to extract text from {pdf_path}: {e}")
    #         return None
    #
    # def download_and_convert(self, papers):
    #     print(f"Downloading PDFs and converting to TXT (max {self.top_n})")
    #     for paper in tqdm(papers):
    #         pdf_path = self.download_pdf(paper)
    #         if pdf_path:
    #             txt_path = self.pdf_to_txt(pdf_path)
    #             if txt_path:
    #                 print(f"Saved: {txt_path}")
    #             else:
    #                 print("PDF to TXT failed.")
    #         else:
    #             print("No open access PDF or download failed.")
def refine_keywords(goal: str, keywords: list) -> list:
    """Refine keywords based on the user goal and existing keywords."""
    prompt = p_manager.render_prompt(
        agent="literature",
        prompt_name="refine_keywords",
        variables={
            "goal": goal,
            "keywords": keywords

        }
    )
    user_input = [{"role": "user", "content": prompt}]
    output, prompt_tokens, completion_tokens = llm_engine.respond(
        user_input, temperature=0.2, top_p=0.92
    )

    # only keywords
    refined = output.split("\n")[0].strip()
    return refined
if __name__ == "__main__":
    a=SemanticScholarHighCiteSearch("spatial clustering validation | spatial cluster validation")
    print(a)
