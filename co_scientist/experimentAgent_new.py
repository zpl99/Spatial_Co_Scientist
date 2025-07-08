from engine.base_engine import LLMEngine
from collections import defaultdict
from litellm import model_cost
from prompt import prompt_manager
import re
from tqdm import tqdm
import itertools

manager = prompt_manager.PromptManager(
    "/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/prompt")

def evaluate_mapeval_classification(dataset):
    total = 0
    correct = 0
    unanswerable = 0
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "unanswerable": 0})

    for item in dataset:
        gt = item['answer']  # ground truth answer id
        pred = item.get('prediction', None)
        category = item['classification']

        if pred is None:
            continue  # skip if prediction is missing

        total += 1
        category_stats[category]['total'] += 1

        if pred == 0:
            unanswerable += 1
            category_stats[category]['unanswerable'] += 1
        if pred == gt:
            correct += 1
            category_stats[category]['correct'] += 1

    overall_accuracy = correct / total if total > 0 else 0.0
    overall_unanswerable_rate = unanswerable / total if total > 0 else 0.0

    per_category_accuracy = {
        cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for cat, stats in category_stats.items()
    }

    per_category_unanswerable_rate = {
        cat: stats["unanswerable"] / stats["total"] if stats["total"] > 0 else 0.0
        for cat, stats in category_stats.items()
    }

    return {
        "overall_accuracy": overall_accuracy,
        "overall_unanswerable_rate": overall_unanswerable_rate,
        "per_category_accuracy": per_category_accuracy,
        "per_category_unanswerable_rate": per_category_unanswerable_rate,
        "total_evaluated": total
    }

class ExperimentDesignAgent:
    def __init__(self, llm_engine_name, context_cutoff=18000):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.llm_cost = model_cost[llm_engine_name]
        self.context_cutoff = context_cutoff
        self.n_generate_sample = 5
        self.prompt_sample = "cct"
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
    def get_samples(self,x,y,n_generate_sample,prompt_sample,stop=None):
        return
    def core_concept_identify(self, problem):
        core_concepts = manager.render_prompt(agent="coreconcepts",prompt_name="core_concepts",variables={})
        prompt = manager.render_prompt(
            agent="coreconcepts",
            prompt_name="coreconcepts_identify",
            variables={}
        )
        user_input = [{"role": "user", "content": prompt}]
        output, *_ = self.llm_engine.respond(user_input, temperature=0.2, top_p=0.92,)
        return output
    def get_votes(self,path):
        return
    def get_values(self):
        return
    def get_answer_id(self, response):
        match = re.search(r'Option id:\s*(\d+)', response)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("No option id found in the response.")

    def cot_solve_task(self, x):

        system_format = manager.render_prompt(
            agent="coreconcepts",
            prompt_name="core_concepts",
            variables={"question": x}
        )
        user_input = [{"role": "user",
                       "content": system_format + '''Identify the relevant core concepts and transformation steps needed to solve it. Think step by step and use transformation path to help you solve the problem. Then, select the correct answer option based on your reasoning. Only output your transformation path. And the selected option's id using the format: Option id: id. Do not include any other explanation. If there is no answer, just output "Option id: 0".'''}]

        result = self.llm_engine.respond(user_input, n=1)[0]
        id = self.get_answer_id(result)
        return id

    def cot_plain_solve_task(self, x):


        user_input = [{"role": "user",
                       "content": x + ''' Think step by step. Only output your reasoning process and your choice of the option's id, in the format of "Option id: id". If you think there is no answer, just output "Option id: 0" to indicate no answer.'''}]

        result = self.llm_engine.respond(user_input, n=1)[0]
        id = self.get_answer_id(result)
        return id

    def generate_next_transformation_prompt(self, question, concepts, current_path,target):
        return manager.render_prompt(
            agent="coreconcepts",
            prompt_name="next_transformation_step",
            variables={
                "question": question,
                "concepts": concepts,
                "current_path": current_path,
                'target': target
            }
        )
    def solve_task(self,x,step):
        ys = []
        infos = []

        system_format = manager.render_prompt(
            agent="coreconcepts",
            prompt_name="core_concepts",
            variables={"question":x}
        )

        user_input = [{"role": "user", "content": system_format+'''Based on the given question, just identify the core concepts involved and also the target core concept (for getting the result), output only: - [id] item (e.g., library): core concept... - [id] target item: core concept'''}]
        # core concepts identification

        core_concepts_result = self.llm_engine.respond(user_input, n=2)[0]
        core_concepts_result = set(core_concepts_result)
        target_concept = 'target item: network'
        for i in core_concepts_result:

            existed_path = []
            # path = set(path[0])
            # values = self.get_votes(path)
            finished = False
            while not finished:
                # generate next transformation step
                prompt = self.generate_next_transformation_prompt(question=x, concepts=i, current_path=existed_path,target=target_concept)
                user_input = [{"role": "user", "content":prompt}]
                result = self.llm_engine.respond(user_input, n=1)[0]
                existed_path.append(result)
                if "stop" in result.lower():
                    finished = True
                #



if __name__ == "__main__":
    import json
    with open("/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/data/mapeval/mapeval_textual.json") as f:
        mapeval_textual = json.load(f)

    item = mapeval_textual[0]
    prompt = (
            "You are a highly intelligent assistant. "
            "Based on the given context, answer the multiple-choice question by selecting the correct option.\n\n"
            "Context:\n" + item["context"] + "\n\n"
                                             "Question:\n" + item["question"] + "\n\n"
                                                                                "Options:\n"
    )
    for i, option in enumerate(item["options"], start=1):
        prompt += f"{i}. {option}\n"
    gt = item["answer"]
    agent = ExperimentDesignAgent("gpt-4o")

    id = agent.solve_task(prompt, step=5)
    item["prediction"] = id
    result = evaluate_mapeval_classification(mapeval_textual)
    print(f"cot cc round", result)

    # for item in tqdm(mapeval_textual):
    #     prompt = (
    #             "You are a highly intelligent assistant. "
    #             "Based on the given context, answer the multiple-choice question by selecting the correct option.\n\n"
    #             "Context:\n" + item["context"] + "\n\n"
    #                                              "Question:\n" + item["question"] + "\n\n"
    #                                                                                 "Options:\n"
    #     )
    #     for i, option in enumerate(item["options"], start=1):
    #         prompt += f"{i}. {option}\n"
    #     gt = item["answer"]
    #     agent = ExperimentDesignAgent("gpt-4o")
    #
    #     id = agent.cot_solve_task(prompt)
    #     item["prediction"] = id
    # result = evaluate_mapeval_classification(mapeval_textual)
    # print(f"cot cc round", result)



    # for item in tqdm(mapeval_textual):
    #     prompt = (
    #             "You are a highly intelligent assistant. "
    #             "Based on the given context, answer the multiple-choice question by selecting the correct option.\n\n"
    #             "Context:\n" + item["context"] + "\n\n"
    #                                              "Question:\n" + item["question"] + "\n\n"
    #                                                                                 "Options:\n"
    #     )
    #     for i, option in enumerate(item["options"], start=1):
    #         prompt += f"{i}. {option}\n"
    #     gt = item["answer"]
    #     agent = ExperimentDesignAgent("gpt-4o")
    #
    #     id = agent.cot_plain_solve_task(prompt)
    #     item["prediction"] = id
    # result = evaluate_mapeval_classification(mapeval_textual)
    # print(f"cot round", result)
