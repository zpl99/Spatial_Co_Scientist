from typing import Callable, List, Tuple
from nano_graphrag import GraphRAG, QueryParam

def rag_expert_input_fn(section, agent_output,graph_func ):

    query = f"{section}: {agent_output}"
    result = graph_func.query(query, param=QueryParam(mode="local", only_need_context=True))
    return result if result else ""

def cli_expert_input_fn(step_name, agent_output):
    print(f" Expert step：{step_name}\n The Agent output：\n{agent_output}")
    feedback = input("Please provide expert opinions or suggestions for revision (press Enter to indicate no revision)：\n")
    return feedback


def expert_in_the_loop(step: str, contexts:str, expert_input_fn:str, history:list, agent_output:str):
    """
    This is a function to interact with an expert in the loop to correct the reasoning process
    step: The step name to be corrected
    contexts: The context to be corrected
    expert_input_fn: The function name you want to use for expert interaction, you can choose: rag_expert_input_fn or cli_expert_input_fn
    """

    if expert_input_fn == "cli_expert_input_fn":
        expert_input_fn = cli_expert_input_fn
        expert_feedback = expert_input_fn(step, agent_output)
    elif expert_input_fn == "rag_expert_input_fn":
        expert_input_fn = rag_expert_input_fn
        expert_feedback = expert_input_fn(step, agent_output)
    else:
        expert_feedback = input(f" [{step}] suggestion:\n")

    if expert_feedback.strip():
        print("use expert recommendation。")
        context = (agent_output, expert_feedback)
    else:
        context = agent_output
    history.append({
        "step": step,
        "agent_output": agent_output,
        "expert_feedback": expert_feedback
        })
    return context, history
