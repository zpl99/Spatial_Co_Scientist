from engine.base_engine import LLMEngine
from litellm import model_cost
from litellm.utils import trim_messages
import requests
import time
from prompt import prompt_manager
from tools import paper_search
SYSTEM_PROMPT = """You are a scientific literature assistant. Your job is to search, summarize, and analyze relevant research papers for a given scientific objective, and provide a chronologically ordered list (from most recent to oldest) of literature and reasoning. 
Your summary should focus on key findings, methods, relevance, and analytical insights, formatted for domain experts."""

p_manager = prompt_manager.PromptManager(
    "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/prompt")

search_engine = paper_search.SemanticScholarHighCiteFetcher("opjXWQ6vW46i5970FzSut2IuJhEWhXLB9Oh28GmK")

class LiteratureAgent:
    def __init__(self, llm_engine_name, context_cutoff=16000, max_results=50):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.llm_cost = model_cost[llm_engine_name]
        self.context_cutoff = context_cutoff
        self.max_results = max_results
        self.history = []
    def search_paper(self,query, max_results=6):
        return search_engine.search_by_query(query, min_citation_count=10,top_n=max_results)

    # def search_arxiv(self, query, max_results=6):
    #     from requests.adapters import HTTPAdapter
    #     from urllib3.util.retry import Retry
    #     import xml.etree.ElementTree as ET
    #
    #     url = (
    #         "http://export.arxiv.org/api/query?"
    #         f"search_query=all:{query.replace(' ', '+')}"
    #         f"&start=0&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending"
    #     )
    #     session = requests.Session()
    #     retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    #     adapter = HTTPAdapter(max_retries=retries)
    #     session.mount('http://', adapter)
    #     session.mount('https://', adapter)
    #
    #     try:
    #         resp = session.get(url, timeout=15)
    #     except requests.exceptions.RequestException as e:
    #         print("arXiv search error:", e)
    #         return []
    #
    #     papers = []
    #     if resp.status_code == 200:
    #         ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    #         root = ET.fromstring(resp.text)
    #         for entry in root.findall('atom:entry', ns):
    #             paper = {}
    #             paper['title'] = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
    #             paper['authors'] = ', '.join(a.text for a in entry.findall('atom:author/atom:name', ns))
    #             paper['published'] = entry.find('atom:published', ns).text[:10]
    #             paper['link'] = entry.find('atom:id', ns).text
    #             paper['summary'] = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
    #             papers.append(paper)
    #     else:
    #         print(f"arXiv API returned status {resp.status_code}")
    #     time.sleep(5)
    #     return papers

    def articles_with_reasoning(self, goal, keywords):
        refine_keywords = self.refine_keywords(goal, keywords)

        search_results = self.search_paper(keywords, self.max_results)

        search_text = "\n\n".join([
            f"[{i + 1}] {paper['title']} ({paper['year']})\nCitationCount: {paper['citationCount']}\nAbstract: {paper['abstract']}"
            for i, paper in enumerate(search_results)
        ])


        SEARCH_PROMPT =  p_manager.render_prompt(
            agent="literature",
            prompt_name="search",
            variables={
                "goal": goal,
                "keywords": refine_keywords,
            }
        )
        prompt =  SEARCH_PROMPT + "\n\nSearch Results:\n" + search_text

        trimmed_prompt = trim_messages(
            [{'role': 'user', 'content': SYSTEM_PROMPT + "\n\n" + prompt}],
            self.llm_engine.llm_engine_name,
            max_tokens=self.context_cutoff
        )[0]['content']

        user_input = [{'role': 'user', 'content': trimmed_prompt}]

        output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.1, top_p=0.95)

        self.history.append({'role': 'user', 'content': trimmed_prompt})

        self.history.append({'role': 'assistant', 'content': output})

        cost = (
                self.llm_cost["input_cost_per_token"] * prompt_tokens +
                self.llm_cost["output_cost_per_token"] * completion_tokens
        )
        return output, cost

    def refine_keywords(self, goal, keywords, n_words=6):
        prompt = p_manager.render_prompt(
            agent="literature",
            prompt_name="refine_keywords",
            variables={
                "goal": goal,
                "keywords": keywords

            }
        )
        user_input = [{"role": "user", "content": prompt}]
        output, prompt_tokens, completion_tokens = self.llm_engine.respond(
            user_input, temperature=0.2, top_p=0.92
        )

        # only keywords
        refined = output.split("\n")[0].strip()
        return refined

    def solve_task(self, task):

        goal = task.get("goal") or task.get("task_inst")
        keywords = task.get("keywords", "spatial clustering")
        output, cost = self.articles_with_reasoning(goal, keywords)
        return {"articles_with_reasoning": output, "cost": cost, "history": self.history}


if __name__ == "__main__":

    agent = LiteratureAgent("gpt-4.1", context_cutoff=14000)

    task = {
        "goal": "How can we make sure that the clusters we generate in complex spatial datasets are both statistically meaningful, and make sense to people who know the area or the problem well?",
        "keywords": "spatial clustering",
    }

    results = agent.solve_task(task)

    print("\n=== Reasoning output ===\n")

    print(results["articles_with_reasoning"])
