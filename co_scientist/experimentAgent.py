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

from prompt import prompt_manager


manager = prompt_manager.PromptManager(
    "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/prompt")

class ExperimentDesignAgent:
    def __init__(self, llm_engine_name, context_cutoff=18000):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.llm_cost = model_cost[llm_engine_name]
        self.context_cutoff = context_cutoff

    def set_experiment_prompt(self, hypothesis, goal, literature="", expert=""):
        rendered_prompt = manager.render_prompt(
            agent="experiment",
            prompt_name="experiment_gen",
            variables={
                "hypothesis": hypothesis,
                "goal": goal,
                "literature": literature,
                "expert": expert,
            }
        )
        return rendered_prompt

    def core_concept_identify(self, hypothesis, goal):
        core_concepts = manager.render_prompt(agent="coreconcepts",prompt_name="core_concepts",variables={})
        prompt = manager.render_prompt(
            agent="coreconcepts",
            prompt_name="coreconcepts_identify",
            variables={"core_concepts": core_concepts,
                       "hypothesis": hypothesis,
                      "goal": goal}
        )
        user_input = [{"role": "user", "content": prompt}]
        output, *_ = self.llm_engine.respond(user_input, temperature=0.2, top_p=0.92)
        return output  # YAML格式或解析后为Python dict/list
    def solve_task(self, hypothesis, goal, literature="", expert=""):
        core_concepts_output = self.core_concept_identify(hypothesis, goal)
        sys_msg = self.set_experiment_prompt(hypothesis, goal, literature, expert)
        user_input = [{'role': 'user', 'content': sys_msg}]
        assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2, top_p=0.95)
        cost = (
            self.llm_cost["input_cost_per_token"] * prompt_tokens +
            self.llm_cost["output_cost_per_token"] * completion_tokens
        )
        return {"experiment_design": assistant_output, "cost": cost}

if __name__ == "__main__":
    agent = ExperimentDesignAgent("gpt-4.1")
    import json
    with open("hypothesis.txt") as f:
        hypothesis_data = json.load(f)

    hypothesis = hypothesis_data['Detailed hypothesis statement']
    goal = hypothesis_data['Restatement of goal and problem if clarifying']
    # preferences = "Aim for academic publication"
    literature = hypothesis_data["Synthesis of literature (chronologically ordered, most recent first)"]
    expert = hypothesis_data["Integration of expert interviews"]
    result = agent.solve_task(hypothesis, goal, literature, expert)
    print(json.dumps(result, indent=2, ensure_ascii=False))
