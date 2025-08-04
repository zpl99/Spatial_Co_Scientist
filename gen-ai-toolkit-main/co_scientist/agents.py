from gait import Agent
import os
from tools.rag_arcgis_doc_tool import ask_doc_ai
from nano_graphrag import GraphRAG, QueryParam
from dotenv import load_dotenv
from literatureAgent import LiteratureAgent

load_dotenv()
graph_func = GraphRAG(working_dir=r"./rag_database",using_azure_openai=True)

def arcgis_doc_interact(query: str) -> str:
    """
    Interact with ArcGIS documentation AI assistant to retrieve relevant information.
    :param query: The query string to ask ArcGIS documentation.
    :return: string containing the relevant information from ArcGIS documentation.
    """
    doc_information = ask_doc_ai(query)
    return doc_information

def paperAgent(goal: str, keywords: str) -> str:

    """
    Interact with the literature agent to retrieve relevant academic papers.
    :param goal: The research goal or question.
    :param keywords: Keywords related to the research topic.
    :return: A string containing the retrieved academic papers and the summary.
    """
    literatureAgent = LiteratureAgent("gpt-4.1")

    return literatureAgent.solve_task({"goal": goal, "keywords": keywords})

def expert_interact(query:str) -> str:
        """
        Interact with experts to gather domain-specific knowledge or context using Graph Rag.
        :param query: The query string to ask experts.
        :return: expert_contex: The context retrieved from expert interaction.
        """
        expert_contex = graph_func.query(query,
            param=QueryParam(mode="local", only_need_context=False) # TODO: expand it
        )

        return expert_contex

def reason_agent(prompt):

    agent = Agent(
        # model="ollama_chat/mistral-small3.1",
        model=f"azure/{os.environ['MODEL']}",
        base_url=f"{os.environ['AZURE_API_URL']}/{os.environ['MODEL']}",
        api_version=os.environ['AZURE_API_VERSION'],
    )
    return agent(prompt)

def expert_agent(prompt):
    instructions = """
    You are an expert agent that provides domain-specific knowledge and context to assist in spatial analysis tasks.
    You run in a loop. At the end of the loop you output an answer.

    Here are the rules you should always follow to solve your task:
    - ALWAYS use the right arguments for the tools. Never use variable names, use the values instead.
    - ALWAYS suffix the final answer with '<Answer/>'
    
    Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000. 

    """
    agent = Agent(
        model=f"azure/{os.environ['MODEL']}",
        base_url=f"{os.environ['AZURE_API_URL']}/{os.environ['MODEL']}",
        instructions=instructions,
        functions=[
            arcgis_doc_interact,
            expert_interact
        ],
        temperature=0.0,
        stop=["<Answer/>"],
    )

    return agent(prompt).agent.messages[2]['content']
