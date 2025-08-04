import requests
import os
from tqdm import tqdm
# from pdfminer.high_level import extract_text
from prompt import prompt_manager
from engine.base_engine import LLMEngine
p_manager = prompt_manager.PromptManager("/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/prompt")

llm_engine = LLMEngine('gpt-4.1')
def SemanticScholarHighCiteSearch(refine_keywords:str, min_citation_count: int = 50,top_n:int = 50, year_from:int = 2015):
    """search high cited paper from Semantic Scholar API based on the user query"""
    refine_keywords = refine_keywords.replace(", ", "|")
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
