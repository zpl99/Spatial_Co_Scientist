from engine.base_engine import LLMEngine

from litellm import model_cost
from litellm.utils import trim_messages
from pathlib import Path
from shutil import copyfile, rmtree
from prompt import prompt_manager

import os
import re
import subprocess

SYSTEM_PROMPT = """You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.
Given a user request, you are expected to write a complete program that accomplishes the requested task and save any outputs in the correct format.
Please wrap your program in a code block that specifies the script type, python. For example:
```python
print("Hello World!")
```"""

SELF_DEBUG_PROMPT = """The user may execute your code and report any exceptions and error messages.
Please address the reported issues and respond with a fixed, complete program."""

FORMAT_PROMPT = """Please keep your response concise and do not use a code block if it's not intended to be executed.
Please do not suggest a few line changes, incomplete program outline, or partial code that requires the user to modify.
Please do not use any interactive Python commands in your program, such as `!pip install numpy`, which will cause execution errors."""

REQUEST_PROMPT = "Here's the user request you need to work on:"

SPATIAL_INFORMATION_PROMPT = """To help you better generate the program, here are useful information. The user's question is related to these spatial core concepts:{concepts} \nThe solution path (using core concepts), its reasoning and the gis functions are here: {transformation_steps}. Please use these to generate the code"""

DATA_INFO_PROMPT = """You can access the dataset at `{dataset_path}`. Here is the directory structure of the dataset:
```
{dataset_folder_tree}
```
Here are some helpful previews for the dataset file(s):
{dataset_preview}"""

manager = prompt_manager.PromptManager(
    "./prompt")


class ScienceAgent():
    def __init__(self, llm_engine_name, context_cutoff=28000, use_self_debug=False, use_knowledge=False):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.llm_cost = model_cost[llm_engine_name]

        self.context_cutoff = context_cutoff
        self.use_self_debug = use_self_debug
        self.use_knowledge = use_knowledge

        self.sys_msg = ""
        self.history = []

    def get_sys_msg(self, task, concepts=None, reasonings=None):
        sys_msg = (
                SYSTEM_PROMPT + "\n\n" +
                (SELF_DEBUG_PROMPT + "\n\n" if self.use_self_debug else "") +
                FORMAT_PROMPT + "\n\n" + REQUEST_PROMPT
        )

        sys_msg += (
                "\n" + task["task_inst"] +
                ("\n" + str(task["domain_knowledge"]) if self.use_knowledge else "") +
                ("\n" + SPATIAL_INFORMATION_PROMPT.format(
                    concepts=concepts,
                    transformation_steps=reasonings
                ) if concepts and reasonings else "")
        )

        sys_msg += (
                "\n" +
                DATA_INFO_PROMPT.format(
                    dataset_path=task['dataset_path'],
                    dataset_folder_tree=task['dataset_folder_tree'],
                    dataset_preview=task["dataset_preview"]
                )
        )

        trimmed_sys_msg = trim_messages(
            [{'role': 'user', 'content': sys_msg}],
            self.llm_engine.llm_engine_name,
            max_tokens=self.context_cutoff - 4000
        )[0]["content"]

        if len(trimmed_sys_msg) < len(sys_msg):
            sys_msg = trimmed_sys_msg + "..."

        return sys_msg

    def write_program(self, assistant_output, out_fname):
        old_program = ""
        if Path(out_fname).exists():
            with open(out_fname, "r", encoding="utf-8") as f:
                old_program = f.read()

        match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = "ERROR"

        with open(out_fname, "w+", encoding="utf-8") as f:
            f.write(result)

        return (old_program == result)  # send early stopping signal if program is unchanged after debugging

    def install(self, out_fname):
        err_msg = ""

        test_path = Path("program_to_eval/")
        if test_path.exists():
            rmtree(test_path)
        os.mkdir(test_path)

        copyfile(out_fname, Path("program_to_eval/", out_fname.split("/")[-1]))

        exec_res = subprocess.run(
            ["pipreqs", "program_to_eval/", "--savepath=requirements.in", "--mode", "no-pin"],
            capture_output=True
        )
        if exec_res.returncode != 0:
            err_msg = "There is a problem extracting packages used in the program. Please use packages that are easier to identify and install via pip."

            return True, err_msg

        exec_res = subprocess.run(
            ["conda", "run", "-n", "sci-agent-eval", "pip-compile", "--upgrade-package", "numpy<2.0", "--resolver",
             "legacy", "--output-file", "eval_requirements.txt"],
            capture_output=True
        )
        if exec_res.returncode != 0:
            print('Legacy resolver failed. Trying backtracking resolver...')
            exec_res = subprocess.run(
                ["conda", "run", "-n", "sci-agent-eval", "pip-compile", "--upgrade-package", "numpy<2.0",
                 "--output-file", "eval_requirements.txt"],
                capture_output=True
            )
            if exec_res.returncode != 0:
                err_msg = "There is a problem resolving the requirements of packages used in the program. Please use packages that do not have conflicts."

                return True, err_msg

        exec_res = subprocess.run(
            ["conda", "run", "-n", "sci-agent-eval", "pip-sync", "eval_requirements.txt"],
            capture_output=True
        )
        if exec_res.returncode != 0:
            err_msg = exec_res.stderr.decode("utf-8")

            trimmed_err_msg = trim_messages(
                [{'role': 'user', 'content': err_msg}],
                self.llm_engine.llm_engine_name,
                max_tokens=2000
            )[0]["content"]

            if len(trimmed_err_msg) < len(err_msg):
                err_msg = trimmed_err_msg + "..."

            return True, err_msg

        return False, err_msg

    def step(self, out_fname, output_fname):
        out_module_name = out_fname.replace("/", '.')[:-3]  # remove ".py" suffix

        special_err, err_msg = self.install(out_fname)

        if not special_err:
            try:
                exec_res = subprocess.run(["conda", "run", "-n", "sci-agent-eval", "python", "-m", out_module_name],
                                          capture_output=True, timeout=900)
            except subprocess.TimeoutExpired:
                special_err = True
                err_msg = "The program fails to finish execution within 900 seconds. Please try to reduce the execution time of your implementation."

        if (not special_err) and exec_res.returncode == 0:
            if not Path(output_fname).exists():
                special_err = True
                err_msg = "The program does not save its output correctly. Please check if the functions are executed and the output path is correct."

        if (not special_err) and exec_res.returncode == 0:
            return True, 0.0
        else:
            if not special_err:
                err_msg = exec_res.stderr.decode("utf-8")

                trimmed_err_msg = trim_messages(
                    [{'role': 'user', 'content': err_msg}],
                    self.llm_engine.llm_engine_name,
                    max_tokens=2000
                )[0]["content"]

                if len(trimmed_err_msg) < len(err_msg):
                    err_msg = trimmed_err_msg + "..."

            user_input = [
                {'role': 'user', 'content': self.sys_msg},
                self.history[-1],
                {'role': 'user', 'content': err_msg}
            ]

            assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2,
                                                                                         top_p=0.95)

            cost = (
                    self.llm_cost["input_cost_per_token"] * prompt_tokens +
                    self.llm_cost["output_cost_per_token"] * completion_tokens
            )

            early_stopping = self.write_program(assistant_output, out_fname)

            self.history += [
                {'role': 'user', 'content': err_msg},
                {'role': 'assistant', 'content': assistant_output}
            ]

            return early_stopping, cost

    def extract_core_concepts(self, task):
        prompt = manager.render_prompt(
            agent="code_generation_core_concept",
            prompt_name="coreconcepts_identify",
            variables={
                "question": task["task_inst"]
            }
        )
        user_input = [{"role": "user", "content": prompt}]
        output, *_ = self.llm_engine.respond(user_input, temperature=0.2, top_p=0.92)
        return output

    def define_transformations(self, task, concepts):
        prompt = manager.render_prompt(
            agent="code_generation_core_concept",
            prompt_name="core_concepts_transformation_path",
            variables={
                "question": task["task_inst"],
                "core_concepts": concepts
            }
        )
        user_input = [{"role": "user", "content": prompt}]
        output, *_ = self.llm_engine.respond(user_input, temperature=0.2, top_p=0.92)
        match = re.search(r'Transformations:\s*(.*)', output, re.DOTALL)
        if not match:
            return []
        transformations_block = match.group(1)

        pattern = r'\[.*?\].*?→.*?(?=(?:,|\n\s*\[|\n*$))'
        matches = re.findall(pattern, transformations_block, re.DOTALL)

        return [m.strip().rstrip(',') for m in matches]

    def reason_transformation_step(self, task, step):
        prompt = manager.render_prompt(agent="code_generation_core_concept", prompt_name="reason_transformation_steps",
                                       variables={"question": task["task_inst"], "transformation_step": step})

        reasoning_output, *_ = self.llm_engine.respond(
            [{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return reasoning_output.strip()

    def transformation_path(self, task):

        concepts = self.extract_core_concepts(task)
        print("Core Concepts identified:", concepts)
        transformation_steps = self.define_transformations(task, concepts)  # Define transformations based on concepts
        reasonings = {}
        for step in transformation_steps:
            print(f"Reasoning transformation step: {step}")
            reasoning = self.reason_transformation_step(task,step)
            print(reasoning)
            reasonings.update({step: reasoning})

        return concepts, reasonings

    def solve_task(self, task, out_fname):
        # Clean history
        self.history = []
        concepts, reasonings = self.transformation_path(task)
        self.sys_msg = self.get_sys_msg(task, concepts, reasonings)

        user_input = [
            {'role': 'user', 'content': self.sys_msg}
        ]

        assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2,
                                                                                     top_p=0.95)

        cost = (
                self.llm_cost["input_cost_per_token"] * prompt_tokens +
                self.llm_cost["output_cost_per_token"] * completion_tokens
        )

        self.write_program(assistant_output, out_fname)

        self.history.append(
            {'role': 'assistant', 'content': assistant_output}
        )

        if self.use_self_debug:
            for t in range(10):
                halt, new_cost = self.step(out_fname, task["output_fname"])
                cost += new_cost
                if halt:
                    break

        self.history = [
                           {'role': 'user', 'content': self.sys_msg}
                       ] + self.history

        return {"history": self.history, "cost": cost}


if __name__ == "__main__":
    agent = ScienceAgent(
        "gpt-4.1",  # "gpt-4o-mini-2024-07-18",
        context_cutoff=28000,
        use_self_debug=False,
        use_knowledge=False
    )


    def format_task_dict(example):
        task = {
            "task_inst": example["task_inst"],
            "dataset_path": example["dataset_folder_tree"].split("\n")[0][4:],
            "dataset_folder_tree": example["dataset_folder_tree"],
            "dataset_preview": example["dataset_preview"],
            "output_fname": example["output_fname"]
        }

        return task


    task = {
        'task_inst': 'Analyze and visualize Elk movements in the given dataset. Estimate home ranges and assess habitat preferences using spatial analysis techniques. Identify the spatial clusters of Elk movements. Document the findings with maps and visualizations. Save the figure as "pred_results/Elk_Analysis.png".',
        'dataset_path': 'benchmark/datasets/ElkMovement/',
        'dataset_folder_tree': '|-- ElkMovement/\n|---- Elk_in_Southwestern_Alberta_2009.geojson',
        'dataset_preview': '[START Preview of ElkMovement/Elk_in_Southwestern_Alberta_2009.geojson]\n{"type":"FeatureCollection","features":[{"type":"Feature","id":1,"geometry":{"type":"Point","coordinates":[-114.19111179959417,49.536741600111178]},"properties":{"OBJECTID":1,"timestamp":"2009-01-01 01:00:37","long":-114.1911118,"lat":49.536741599999999,"comments":"Carbondale","external_t":-5,"dop":2.3999999999999999,"fix_type_r":"3D","satellite_":0,"height":1375.1900000000001,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230771637000,"summer_indicator":1}},{"type":"Feature","id":2,"geometry":{"type":"Point","coordinates":[-114.1916239994119,49.536505999952517]},"properties":{"OBJECTID":2,"timestamp":"2009-01-01 03:00:52","long":-114.191624,"lat":49.536506000000003,"comments":"Carbondale","external_t":-6,"dop":2.3999999999999999,"fix_type_r":"3D","satellite_":0,"height":1375.2,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230778852000,"summer_indicator":1}},{"type":"Feature","id":3,"geometry":{"type":"Point","coordinates":[-114.19169140075056,49.536571800069581]},"properties":{"OBJECTID":3,"timestamp":"2009-01-01 05:00:49","long":-114.1916914,"lat":49.536571799999997,"comments":"Carbondale","external_t":-6,"dop":5.6000000000000014,"fix_type_r":"3D","satellite_":0,"height":1382.0999999999999,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230786049000,"summer_indicator":1}},...]}\n[END Preview of ElkMovement/Elk_in_Southwestern_Alberta_2009.geojson]',
        'output_fname': 'pred_results/Elk_Analysis.png'}
    task = format_task_dict(task)
    ut_fname = Path( 'pred_programs/', "pred_" + "elk_new.py")
    trajectory = agent.solve_task(task, out_fname=str(ut_fname))
    print(trajectory)
