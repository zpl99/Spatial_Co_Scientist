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

    def refine_hypothesis(self, draft_text, round_idx=0):
        """
        multi rounds of hypothesis refinement with expert feedback.
        """
        current_text = draft_text
        # 1. log the draft response
        self.history.append({
            "role": "llm_draft",
            "content": current_text,
            "meta": {"round": round_idx}
        })

        # 2. judge whether expert intervention is needed
        expert_feedback = expert_interact_workflow(current_text,self.history,self.llm_engine)
        self.history.append({
            "role": "expert_review",
            "content": expert_feedback,
            "meta": {"round": round_idx}
        })

        # 3. 合成prompt，带入所有历史，交给LLM重新输出
        merge_prompt = (
            "Below is the latest hypothesis draft and expert feedback history.\n"
            "Your job is to integrate the expert feedback into the hypothesis, "
            "refining and improving the text for publication, ensuring clarity, scientific rigor, and coherence. "
            "If any expert suggestions conflict, use your best judgment as a domain expert to resolve.\n\n"
            "====\n"
            "History (from earliest to latest):\n"
        )
        history = f"Round {round_idx}:\n Role: {self.history[-1]['role']}\n Content: {self.history[-1]['content']}\n"
        merge_prompt += history

        merge_prompt += "\n---\nOutput the fully revised hypothesis below (since you have revised the overall process, so here you just output the final revised hypothesis, do not include other content):\n"

        # 4. generate new hypothesis
        revised_text, _, _ = self.llm_engine.respond(
            [{'role': 'user', 'content': merge_prompt}],
            temperature=0.2, top_p=0.95
        )

        self.history.append({
            "role": "llm_refined",
            "content": revised_text,
            "meta": {"round": round_idx}
        })


        return revised_text



    def solve_task(self, task):

        # Clean history
        self.history = []

        self.sys_msg = self.set_hypothesis_prompt(task)

        self.history.append({"role": "system", "content": self.sys_msg, "meta": {}})

        user_input = [
            {'role': 'user', 'content': self.sys_msg}
        ]

        assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2,
                                                                                     top_p=0.95)


        final_hypothesis = self.refine_hypothesis(assistant_output, round_idx=0)
        print(final_hypothesis)
        self.save_formatted_output(final_hypothesis,"./")

        self.history.append(
            {'role': 'assistant', 'content': assistant_output}
        )

        self.history = [
                           {'role': 'user', 'content': self.sys_msg}
                       ] + self.history

        return {"history": self.history,}


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
