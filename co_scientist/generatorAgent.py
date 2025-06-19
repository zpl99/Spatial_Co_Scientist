from engine.base_engine import LLMEngine

from litellm import model_cost
from litellm.utils import trim_messages
from pathlib import Path
from shutil import copyfile, rmtree

import os
import re
import subprocess
from literatureAgent import LiteratureAgent
import json
from prompt import prompt_manager
from nano_graphrag import GraphRAG, QueryParam
from expert_interact_agent import expert_interact_workflow

manager = prompt_manager.PromptManager(
    "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/prompt")

graph_func = GraphRAG(working_dir="/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/rag_database",using_azure_openai=True)


literatureAgent = LiteratureAgent("gpt-4.1")


class GeneratorAgent():
    def __init__(self, llm_engine_name, context_cutoff=28000, use_rag=True, use_literature=True):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.llm_cost = model_cost[llm_engine_name]

        self.context_cutoff = context_cutoff
        self.use_rag = use_rag
        self.use_literature = use_literature

        self.sys_msg = ""
        self.history = []

    def save_formatted_output(self, assistant_output, save_dir, fname_base="hypothesis"):

        os.makedirs(save_dir, exist_ok=True)
        clean_output = re.sub(r'\n\s*\n', '\n\n', assistant_output.strip())

        txt_path = os.path.join(save_dir, f"{fname_base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_output)
        # 写json（每一节单独为一项，便于后续处理）
        print(f"Assistant output saved to:\n  {txt_path}")

    def literature_review(self, task):
        return literatureAgent.solve_task(task)

    def set_hypothesis_prompt(self, task):
        literature_results = self.literature_review(task) if self.use_literature else ""

        expert_contex = graph_func.query(
            task.get("goal"),
            param=QueryParam(mode="local", only_need_context=True)
        ) if self.use_rag else ""
        rendered_prompt = manager.render_prompt(
            agent="generator",
            prompt_name="hypothesis",
            variables={
                "goal": task.get("goal"),
                "preferences": task.get("preferences", ""),
                "source_hypothesis": task.get("source_hypothesis", ""),
                "articles_with_reasoning": literature_results,
                "expert_contexts": expert_contex,

            }
        )
        trimmed_sys_msg = trim_messages(
            [{'role': 'user', 'content': rendered_prompt}],
            self.llm_engine.llm_engine_name,
            max_tokens=self.context_cutoff - 2000
        )[0]["content"]

        if len(trimmed_sys_msg) < len(rendered_prompt):
            rendered_prompt = trimmed_sys_msg + "..."

        return rendered_prompt
    def refine_hypothesis(self, task):
        while True:
            self.sys_msg = self.set_hypothesis_prompt(task)

            user_input = [
                {'role': 'user', 'content': self.sys_msg}
            ]

            assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2,
                                                                                         top_p=0.95)
            self.save_formatted_output(assistant_output, "./", fname_base="hypothesis")

            cost = (
                    self.llm_cost["input_cost_per_token"] * prompt_tokens +
                    self.llm_cost["output_cost_per_token"] * completion_tokens
            )

            self.history.append(
                {'role': 'assistant', 'content': assistant_output}
            )

            self.history = [
                               {'role': 'user', 'content': self.sys_msg}
                           ] + self.history

            expert_feedback = expert_interact_workflow(
                "Hypothesis",self.history,
                assistant_output,
                graph_func,
            )
            if not expert_feedback.strip():
                break
            else:
                task["source_hypothesis"] = expert_feedback



    def solve_task(self, task):

        # Clean history
        self.history = []

        self.sys_msg = self.set_hypothesis_prompt(task)

        user_input = [
            {'role': 'user', 'content': self.sys_msg}
        ]

        assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2,
                                                                                     top_p=0.95)
        self.save_formatted_output(assistant_output,"./")
        cost = (
                self.llm_cost["input_cost_per_token"] * prompt_tokens +
                self.llm_cost["output_cost_per_token"] * completion_tokens
        )

        self.history.append(
            {'role': 'assistant', 'content': assistant_output}
        )

        self.history = [
                           {'role': 'user', 'content': self.sys_msg}
                       ] + self.history

        return {"history": self.history, "cost": cost}


if __name__ == "__main__":

    agent = GeneratorAgent(
        "gpt-4.1",  # "gpt-4o-mini-2024-07-18",
        context_cutoff=28000,
        use_rag=True,
        use_literature=True
    )

    task = {
        "goal":"How can we make sure that the clusters we generate in complex spatial datasets are both statistically meaningful, and make sense to people who know the area or the problem well?",
        "preferences":"supported for academic publication",
        "key_words":"spatial clustering"
    }

    trajectory = agent.solve_task(task)
