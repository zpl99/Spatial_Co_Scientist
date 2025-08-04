import os
import logging
import warnings
def silence_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.disabled = True

for name in ["httpx", "httpcore", "nano-graphrag","nano-vectordb","LiteLLM"]:
# for name in ["httpx", "httpcore", "LiteLLM"]:
    silence_logger(name)

from gait import Node, Flow, ScratchpadInMemory, Agent, FlowObserverConsole
from guidedAnalysis import GuidedAnalysisSystem
from agents import expert_interact, arcgis_doc_interact, paperAgent
from tools import read_data
from nano_graphrag import QueryParam
from prompt import prompt_manager
import textwrap
from colorama import Fore, Style, init

p_manager = prompt_manager.PromptManager(r"./prompt")
react_tools = GuidedAnalysisSystem(max_turns=50)


def create_agent():
    return Agent(
        model=f"azure/{os.environ['MODEL']}",
        base_url=f"{os.environ['AZURE_API_URL']}/{os.environ['MODEL']}",
        api_version=os.environ['AZURE_API_VERSION'],
        max_tokens=20000,
    )

reason_agent = create_agent()
summary_agent = create_agent()


def append_history(sp, role, content):
    sp['history'].append({'role': role, 'content': content})


# === Nodes ===
class AgentStartNode(Node):
    def exec(self, sp):
        sp['history'] = react_tools.start_session(sp['user_input'])
        sp['user_information'] = [{'role': 'user', 'content': sp['user_input']}]
        sp["input_action_query_obs_pair"] = []
        sp['turn'] = 0
        return "data_init" if sp.get("data_layer") else self.DEFAULT

class AgentReasonNode(Node):
    def exec(self, sp):
        agent_reply = reason_agent(sp['history']).content
        action, input_query = react_tools.react_decision(agent_reply)
        logging.info(f"Action: {action}, Input: {input_query}")

        data_prompt = p_manager.render_prompt(
            agent="data_analysis",
            prompt_name="data_analysis",
            variables={"user_in": sp["user_input"], "background": sp["data_background"]}
        )

        result = reason_agent(data_prompt).content

        if result != "stop":
            sp["data_prompt"] = result
            return "data_reason"

        sp['user_information'] = [sp['user_information'][0]]
        sp["input"] = input_query
        sp["last_action"] = action
        append_history(sp, 'assistant', agent_reply)

        return action if action in [
            "ask_user", "talk_expert", "arcgis_pro_document_retrieval",
            "literature_search", "finish"
        ] else self.DEFAULT


class UserInputNode(Node):
    def exec(self, sp):
        question_prompt = p_manager.render_prompt(agent="guidedanalysis", prompt_name="ask_user", variables={"context": sp["input_action_query_obs_pair"],"question_info": {"user question": sp['user_input'],"system follow up question":sp["input"]},"data_info":sp['data_background']})

        agent_reply = reason_agent(question_prompt).content
        print(agent_reply)
        print(Fore.RED+"\n Question"+ Style.RESET_ALL, textwrap.fill(sp["input"], width=120))

        sp["input_action_query_obs_pair"] = []
        user_input = input("Your answer or clarification: ")
        append_history(sp, 'system display', agent_reply)
        append_history(sp, 'user', user_input)
        sp['user_input'] = user_input
        return self.DEFAULT


class DataInitialNode(Node):
    def exec(self, sp):
        logging.info("Performing data initialization...")
        rag_query_function, obs = read_data.generate_llm_context_from_layer_json(sp["data_layer"])
        prompt = p_manager.render_prompt("data_analysis", "background_summary", {"raw_obs": obs})
        background_info = reason_agent(prompt).content
        sp.rag_query_function = rag_query_function
        sp["data_background"] = background_info
        append_history(sp, 'system', f"This is the information about the user data, keep in mind of it:\n{background_info}")
        return self.DEFAULT


class DataReasonNode(Node):
    def exec(self, sp):
        try:
            result = sp.rag_query_function.query(sp['data_prompt'], param=QueryParam(mode='local'))

        except Exception as e:
            result = f"Query failed: {e}"

        message = f"Prompt: {sp['data_prompt']}. New info: {result}"
        sp['data_background'] += "\n" + message

        for entry in sp["history"]:
            if entry['role'] == 'system' and entry['content'].startswith("This is the information"):
                entry['content'] = f"This is the information about the user data, keep in mind of it:\n{sp['data_background']}"
                break
        else:
            append_history(sp, 'system', f"This is the information about the user data, keep in mind of it:\n{sp['data_background']}")
        sp["input_action_query_obs_pair"].append(("Data Reason",sp['data_prompt'],result))
        return self.DEFAULT


class ExpertKnowledgeInteractNode(Node):
    def exec(self, sp):
        sp['observation'] = expert_interact(sp['input'])
        append_history(sp, 'system observation', sp['observation'])
        sp["input_action_query_obs_pair"].append(("Expert Knowledge", sp['input'], sp['observation']))
        return self.DEFAULT


class ArcGISDocInteractNode(Node):
    def exec(self, sp):
        sp['observation'] = arcgis_doc_interact(sp['input'])
        append_history(sp, 'system observation', sp['observation'])
        sp["input_action_query_obs_pair"].append(("Arcgis doc info", sp['input'], sp['observation']))
        return self.DEFAULT


class LiteratureSearchNode(Node):
    def exec(self, sp):
        sp['observation'] = paperAgent(sp['input']['goal'], sp['input']['keywords'])
        append_history(sp, 'system observation', sp['observation'])
        return self.DEFAULT


class FinishNode(Node):
    def exec(self, sp):
        logging.info("Analysis complete.")
        sp['finished'] = True
        return self.DEFAULT


# === Main Flow ===
if __name__ == "__main__":
    start = AgentStartNode()
    reason = AgentReasonNode()
    userin = UserInputNode()
    expert = ExpertKnowledgeInteractNode()
    paper = LiteratureSearchNode()
    arcgisai = ArcGISDocInteractNode()
    datainit = DataInitialNode()
    datareason = DataReasonNode()
    finish = FinishNode()

    start >> reason
    start - "data_init" >> datainit >> reason
    reason - "data_reason" >> datareason >> reason
    reason - "ask_user" >> userin >> reason
    reason - "talk_expert" >> expert >> reason
    reason - "arcgis_pro_document_retrieval" >> arcgisai >> reason
    # reason - "literature_search" >> paper >> reason
    reason - "finish" >> finish

    flow = Flow(start, observer=FlowObserverConsole())
    sp = ScratchpadInMemory()
    user_input = input("Your question: ")
    flow(user_input=user_input, data_layer="patth/to_which")
    flow()