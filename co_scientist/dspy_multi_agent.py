import dspy
from tools.paper_search import SemanticScholarHighCiteSearch, refine_keywords
from tools.knowledge_rag import get_knowledge_rag
from tools.expert_interact import rag_expert_input_fn, cli_expert_input_fn, expert_in_the_loop
class PaperSearch(dspy.Signature):
    """You are a literature review agent, given the user request, you are asked to firstly refine the keywords based on the user request, and then call the function to search for relevant paper.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fulfill users' request. Please search as many papers as possible so that the user can get a comprehensive understanding of the topic."""

    goal: str = dspy.InputField()
    keywords: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Please summarize the most relevant articles (preferably 3-8), ordered by year, each with citation info, a 2-4 sentence summary, and your analysis of how it relates to the objective. "
        )
    )

class HypothesisGeneration(dspy.Signature):
    """You are a hypothesis generation agent, given the user request, you are asked to generate a hypothesis based on the user request.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fullfil users' request. Please generate a hypothesis that is testable and can be used for further research."""

    goal: str = dspy.InputField()
    preferences: str = dspy.InputField()
    generated_hypothesis: str = dspy.OutputField(
        desc=''' Your output hypothesis should include each of the following, in order:
    - Restatement of goal and problem if clarifying.
    - Reasoning and analysis steps (step by step).
    - Synthesis of literature (chronologically ordered, most recent first).
    - Logical framework/rationale connecting literature and hypothesis.
    - Detailed hypothesis statement.
    - Brief mapping of how the hypothesis satisfies each criterion.'''
    )

if __name__ == "__main__":

    agent = dspy.ReAct(
        HypothesisGeneration,
        tools=[
            # SemanticScholarHighCiteSearch,
            # refine_keywords,
            # get_knowledge_rag,
            rag_expert_input_fn,
            cli_expert_input_fn,
            expert_in_the_loop
        ],
        max_iters=10
    )

    lm = dspy.LM('azure/gpt-4.1', api_key='3c744295edda4f13bf0ef198ddb4d24c',
                 api_base='https://ist-apim-aoai.azure-api.net/load-balancing/gpt-4.1')

    dspy.configure(lm=lm)

    result = agent(goal="How can we make sure that the clusters we generate in complex spatial datasets are both statistically meaningful, and make sense to people who know the area or the problem well?", preference="supported for academic publication")

    print(result)