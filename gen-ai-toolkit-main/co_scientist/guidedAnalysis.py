from sqlalchemy.dialects.postgresql import JSONB

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
from tools.rag_arcgis_doc_tool import ask_doc_ai

manager = prompt_manager.PromptManager(
    r"./prompt")

graph_func = GraphRAG(working_dir=r"./rag_database",using_azure_openai=True)

class GuidedAnalysisSystem:
    def __init__(self,max_turns=6):
        self.system_prompt = manager.render_prompt(agent = "guidedanalysis", prompt_name="system_prompt", variables={})
        self.max_turns = max_turns

    def set_display_prompt(self, contexts):
        return manager.render_prompt(agent = "guidedanalysis", prompt_name="ask_user", variables={"context": contexts})

    def start_session(self, initial_user_input):
        history = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': initial_user_input}
        ]
        return history

    def get_user_visible_message(self, agent_reply, action, ask_user_flag, observation=None):
        """
        Generate the message suitable for the end user, based on the current agent action and context.
        - For ask_user: Show the clarifying question directly.
        - For finish: Show the agent's summary/final recommendation.
        - For tool actions (literature_search, expert_knowledge_interact, arcgis_document_retrieval):
            Display a friendly status and a summary of retrieved results, if available.
        - For unknown actions: Show a default message.
        This keeps LLM's internal reasoning and thoughts hidden, giving the user a clean, guided interaction.

        Args:
            agent_reply (str): The agent's latest response (may contain system prompts, thoughts, etc).
            action (str): The extracted action keyword.
            ask_user_flag (bool): If the agent expects user input/clarification.
            observation (Any): Tool result or retrieved content to summarize to the user.

        Returns:
            str or None: The message to display to the user, or None if nothing to show.
        """
        # 1. If the agent is prompting the user for more information, show the clarifying question.
        if ask_user_flag:
            # Typically, agent_reply is the clarifying question itself.
            return agent_reply.strip()

        # 2. If the agent has reached the end of the session (finish action), show summary/closing statement.
        if action == "finish":
            return agent_reply.strip()

        # 3. Tool calls: Display helpful tool status and summarized results.
        if action == "literature_search":
            tip = "The system is reviewing relevant academic literature for you."
            # Optionally show part of the literature summary, but keep it concise.
            if observation:
                if isinstance(observation, dict) and "articles_with_reasoning" in observation:
                    summary = observation["articles_with_reasoning"]
                    if len(summary) > 500:
                        summary = summary[:500] + "..."
                    tip += "\n\nSummary:\n" + summary
                elif isinstance(observation, str):
                    tip += "\n\nSummary:\n" + (observation[:500] + "..." if len(observation) > 500 else observation)
            return tip

        if action == "expert_knowledge_interact":
            tip = "The system has consulted domain expert knowledge."
            if observation:
                if isinstance(observation, str) and len(observation) > 0:
                    summary = observation[:500] + ("..." if len(observation) > 500 else "")
                    tip += "\n\nExpert insight:\n" + summary
            return tip

        if action == "arcgis_document_retrieval":
            tip = "The system has looked up ArcGIS Pro documentation for your question."
            if observation:
                summary = observation[:500] + ("..." if len(observation) > 500 else "")
                tip += "\n\nDocumentation result:\n" + summary
            return tip

        # 4. Fallback for unknown or unexpected actions.
        if action == "unknown action" or action is None:
            print(agent_reply)
            return "The system has received your input and is processing the next step..."

        # 5. Default: return nothing.
        return None

    def react_decision(self, agent_thought):
        """
        Parse the agent's 'Thought/Action/Input' and call the corresponding tool, or decide to ask user.
        Supported actions: literature_search, expert_knowledge_interact, arcgis_document_retrieval, ask_user, finish
        Returns: (action, observation, ask_user_flag)
        """
        json_block = None
        match_json = re.search(r'Action\s*:\s*(\{.*\})', agent_thought, re.DOTALL)
        if match_json:
            try:
                json_block = json.loads(match_json.group(1))
            except Exception as e:
                print(f"[react_decision] JSON parse failed: {e}")
                json_block = None

        if json_block:
            action = json_block.get("action")
            input_query = json_block.get("input")
        else:
            match = re.search(r"Action\s*:\s*(\w+)\s*;?\s*Input\s*:\s*(.+)", agent_thought, re.I)
            if not match:
                return None, None, False
            action, input_query = match.group(1).strip().lower(), match.group(2).strip()

        return action, input_query

    def next_turn(self, history):

        response, _, _ = self.llm_engine.respond(history, temperature=0.2, top_p=0.95)
        return response

    def run_guided_analysis(self, initial_user_input):
        history = self.start_session(initial_user_input)
        for turn in range(self.max_turns):
            agent_reply = self.next_turn(history)
            print(f"\n[Agent]: {agent_reply.strip()}\n")

            action, observation, ask_user_flag = self.react_decision(agent_reply)

            while action and not ask_user_flag and action != "finish":
                obs_msg = f"Observation: {observation}"
                print(f"\n[Tool]: {obs_msg}\n")
                history.append({'role': 'assistant', 'content': agent_reply})
                history.append({'role': 'system', 'content': obs_msg})

                agent_reply = self.next_turn(history)
                print(f"\n[Agent]: {agent_reply.strip()}\n")
                action, observation, ask_user_flag = self.react_decision(agent_reply)

            if ask_user_flag:
                history.append({'role': 'assistant', 'content': agent_reply})
                user_input = input("[User]: ")
                history.append({'role': 'user', 'content': user_input})
                if self._should_terminate(agent_reply, user_input, turn):
                    break
                continue

            if action == "finish":
                history.append({'role': 'assistant', 'content': agent_reply})
                print("[Agent]: Analysis finished.")
                break

            if not action:
                history.append({'role': 'assistant', 'content': agent_reply})
                user_input = input("[User]: ")
                history.append({'role': 'user', 'content': user_input})
                if self._should_terminate(agent_reply, user_input, turn):
                    break

        return history

    def _should_terminate(self, agent_reply, user_input, turn):
        if "enough" in user_input.lower() or "ready to proceed" in agent_reply.lower():
            return True
        if turn >= self.max_turns - 1:
            print("\n[Agent]: Maximum turns reached. Ending session.")
            return True
        return False



if __name__ == "__main__":

    llm_engine = LLMEngine("gpt-4.1")
    agent = GuidedAnalysisSystem(llm_engine)
    # user_input = input("Describe your spatial analysis need (e.g., 'I want to find retail clusters'), enter enough to stop this session: ")
    user_input = "i want to find retail clusters in Austin, Texas."

    history = agent.run_guided_analysis(user_input)

    print("\n==== Conversation History ====")
    for turn in history:
        print(f"{turn['role']}: {turn['content']}")
