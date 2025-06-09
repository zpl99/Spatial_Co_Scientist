# from engine.base_engine import LLMEngine

from litellm import model_cost
from litellm.utils import trim_messages
from pathlib import Path
from shutil import copyfile, rmtree
from engine.openai_engine import MyAzureOpenaiEngine
import geopandas as gpd
import os
import re
import subprocess
import irs_template_2

SYSTEM_PROMPT = f"""You are a spatial clustering expert agent, designed to help scientists structure their analysis needs into a formal Intermediate Representation Schema (IRS).
Your goal is to convert user spatial clustering requests into a structured, executable IRS in JSON format, which can later be translated by an interpreter into real GIS code (e.g., using GeoPandas, ArcPy, or SQL). For example:
```You must:
1. Understand the spatial clustering intent from the user prompt.
2. Try to get some background knowledge from the user about the dataset and the clustering task. You can search the website to get the result.
3. Identify the key elements (e.g., clustering method, variables, spatial constraints).
4. Construct a complete and valid JSON representing the IRS specification.
5. Include multiple steps in the pipeline if necessary and use `step_id` and `dep` to describe dependencies. 

All JSON should be embedded inside a Python code block with this pattern, here is a template, please use this format but change the irs content:
```python
irs = {irs_template_2.irs_template}
```"""


SELF_DEBUG_PROMPT = """If the user executes your code and reports any exception or parsing error (e.g., invalid JSON structure, missing field, logical inconsistency), your task is to correct it and regenerate the complete, fixed IRS.
Respond only with a full replacement program. Avoid partial edits or explanations."""


FORMAT_PROMPT = """Please output a complete and valid Python script containing the IRS as a variable.
Do not wrap the IRS in text explanations or notes.
Do not provide partial outlines or ask the user to fill in any information manually.
Avoid interactive or shell commands (e.g., `!pip install`) that will raise execution errors in downstream use."""


REQUEST_PROMPT = "Here's the user request you need to work on:"


# DATA_INFO_PROMPT = """You can access the dataset at `{dataset_path}`. Here is the directory structure of the dataset:
# ```
# {dataset_folder_tree}
# ```
# Here are some helpful previews for the dataset file(s):
# {dataset_preview}"""

DATA_INFO_PROMPT = """You can access the dataset at `{dataset_path}`. 
Here are some helpful previews for the dataset file(s):
{dataset_preview}"""

WORK_FLOW_PROMPT = ("These are some potential method you may want to use (please also consider others this is just for reference): "
                    "1. Build Balanced Zones: Creates spatially contiguous zones in a study area using a genetic growth algorithm based on specified criteria. "
                    "2. Calculate Composite Index: Combines multiple numeric variables to create a single index. "
                    "3. Cluster and Outlier Analysis (Anselin Local Moran's I): Identifies statistically significant hot spots, cold spots, and spatial outliers using the Anselin Local Moran's I statistic, given a set of weighted features. "
                    "4. Density-based Clustering: Finds clusters of point features within surrounding noise based on their spatial distribution. Time can also be incorporated to find space-time clusters. "
                    "5. Hot Spot Analysis (Getis-Ord Gi*): Given a set of weighted features, identifies statistically significant hot spots and cold spots using the Getis-Ord Gi* statistic. "
                    "6. Hot Spot Analysis Comparison: Compares two hot spot analysis result layers and measures their similarity and association. "
                    "7. Multivariate Clustering: Finds natural clusters of features based solely on feature attribute values. "
                    "8. Optimized Hot Spot Analysis: Creates a map of statistically significant hot and cold spots using the Getis-Ord Gi* statistic, given incident points or weighted features (points or polygons). "
                    "9. Optimized Outlier Analysis: Given incident points or weighted features (points or polygons), creates a map of statistically significant hot spots, cold spots, and spatial outliers using the Anselin Local Moran's I statistic. It evaluates the characteristics of the input feature class to produce optimal results."
                    "10. Similarity Search: Identifies which candidate features are most similar or most dissimilar to one or more input features based on feature attributes."
                    "11. Spatial Outlier Detection：Identifies global or local spatial outliers in point features."
                    "12. Spatially Constrained Multivariate Clustering：Finds spatially contiguous clusters of features based on a set of feature attribute values and optional cluster size limits.  ")

class ScienceAgent():
    def __init__(self, engine, context_cutoff=28000, use_self_debug=False, use_knowledge=False):
        self.llm_engine = engine
        self.llm_cost = model_cost.get(engine.model_name, {"input_cost_per_token": 0, "output_cost_per_token": 0})

        self.context_cutoff = context_cutoff
        self.use_self_debug = use_self_debug
        self.use_knowledge = use_knowledge

        self.sys_msg = ""
        self.history = []

    def parse_shapefile_attributes_with_descriptions(self, shapefile_path):
        """
        read shapefile and parse its attributes with descriptions using LLM.
        """
        try:
            gdf = gpd.read_file(shapefile_path)
        except Exception as e:
            return f"Error reading shapefile: {e}"

        attribute_info = []
        for col in gdf.columns:
            if col != gdf.geometry.name:
                dtype = str(gdf[col].dtype)
                attribute_info.append({"name": col, "type": dtype})

        description_prompt = "Please provide a short but informative, and plain-English explanation for the following GIS attributes:\n"
        for attr in attribute_info:
            description_prompt += f"- {attr['name']} ({attr['type']}):\n"

        messages = [{"role": "user", "content": description_prompt}]
        assistant_output, *_ = self.llm_engine.respond(messages, temperature=0.2, top_p=0.95)

        # 合并返回结果
        preview_block = "Attributes with descriptions:\n"
        preview_block += assistant_output.strip()

        return preview_block
    def get_sys_msg(self, task):
        sys_msg = (
                SYSTEM_PROMPT + "\n\n" +
                (SELF_DEBUG_PROMPT + "\n\n" if self.use_self_debug else "") +
                FORMAT_PROMPT + "\n\n" + REQUEST_PROMPT
        )
        sys_msg += (
            "[TASK DESCRIPTION]" + task["task_inst"]
        )
        dataset_preview = self.parse_shapefile_attributes_with_descriptions(task['dataset_path'])
        sys_msg += (
                "[DATA INFO]" +
                DATA_INFO_PROMPT.format(
                    dataset_path=task['dataset_path'],
                   #  dataset_folder_tree=task['dataset_folder_tree'],
                    dataset_preview=dataset_preview
                )
        )

        trimmed_sys_msg = trim_messages(
            [{'role': 'user', 'content': sys_msg}],
            self.llm_engine.model_name,
            max_tokens=self.context_cutoff - 2000
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
    def generate_code_from_irs(self,irs,history, output_file):
        """
        Generate code from IRS using the LLM engine.
        """
        history_content = "\n\n".join(
            [f"{item['role'].upper()}:\n{item['content']}" for item in history]
        )

        code_gen_prompt = f"""
        You are an expert spatial clustering Python developer. Your task is to convert the following Intermediate Representation Schema (IRS) into executable Python code.  
        Here is the full conversation history for context:
        {history_content}.
        Here is the IRS specification:
        ```python
        {irs}
        Please generate Python code to fully implement the spatial clustering workflow described in this IRS using GeoPandas and scikit-learn.
        
        Your Python script should include:

        Data loading and preprocessing (handle missing values, normalization, multicollinearity removal)

        Clustering analysis steps as specified in the pipeline

        Evaluation and visualization of clustering results based on the evaluation plan

        The output should be self-contained and ready for execution.

        Output the complete Python script wrapped in a Python code block (python ... ).
        
        Please do map_visualization, use a map that plots the clusters in geospatial terms.
        
        Also, please print some statistical information about the clustering results, such as cluster sizes, centroids, and any other relevant metrics for user to understand the model output
        
        Do not leave blank or placeholder in the code. All the code should be executable and complete. Please think carefully before you write the code, and make sure the code is correct and executable.
        """

        messages = [{"role": "user", "content": code_gen_prompt}]

        assistant_output, *_ = self.llm_engine.respond(messages, temperature=0.2, top_p=0.95)

        match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)

        if match:
            generated_code = match.group(1).strip()
        else:
            generated_code = "# ERROR: Code extraction failed."

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_code)

        return generated_code

    def solve_task_ablation_direct_code(self, task, out_fname):
        """
        Bypasses IRS and directly generates executable spatial clustering code from user task.
        """
        self.history = []

        # 提取数据预览信息
        dataset_preview = self.parse_shapefile_attributes_with_descriptions(task['dataset_path'])

        # 构造 prompt
        direct_code_prompt = f"""
    You are a spatial clustering Python expert.
    Your task is to write complete and executable Python code that solves the following user question using spatial data and clustering techniques.

    [TASK DESCRIPTION]
    {task['task_inst']}

    [DATA INFO]
    You can access the dataset at `{task['dataset_path']}`.
    Here is a preview of the dataset attributes:
    {dataset_preview}

    Your output should be a full Python script that includes:
    - Data loading
    - Preprocessing (e.g., missing values, normalization)
    - Clustering analysis
    - Result visualization or interpretation (e.g., plots or print statements)

    Please output only Python code wrapped in a ```python code block. Do not leave placeholders.
    """

        messages = [{"role": "user", "content": direct_code_prompt}]
        assistant_output, *_ = self.llm_engine.respond(messages, temperature=0.2, top_p=0.95)

        self.history.append({'role': 'user', 'content': direct_code_prompt})
        self.history.append({'role': 'assistant', 'content': assistant_output})

        match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = "# ERROR: Failed to extract Python code block."

        with open(out_fname, "w", encoding="utf-8") as f:
            f.write(code)

        return {"history": self.history, "output_file": out_fname}
    def solve_task(self, task, out_fname):
        # Clean history
        print("Generate IRS...")
        self.history = []

        self.sys_msg = self.get_sys_msg(task)

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


        match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
        if match:
            irs_code = match.group(1).strip()
            exec(irs_code, globals())
            generated_irs = globals().get("irs", None)
        else:
            generated_irs = None
        print("Generate Code using interpreter...")
        if generated_irs:
            interpreter_output_path = out_fname.replace(".py", "_interpreted.py")
            self.generate_code_from_irs(generated_irs, self.history, interpreter_output_path)

        if self.use_self_debug:
            for t in range(10):
                halt, new_cost = self.step(out_fname, task["output_fname"])
                cost += new_cost
                if halt:
                    break

        self.history = [{'role': 'user', 'content': self.sys_msg}] + self.history

        return {"history": self.history, "cost": cost}


if __name__ == "__main__":
    engine = MyAzureOpenaiEngine(
        api_key="192cd313f4bb4a31b0fbba21a18f8c1a",
        api_version="2024-10-01-preview",
        azure_endpoint="https://seai.openai.azure.com/",
        model_name="gpt-4o"
    )

    agent = ScienceAgent(
        engine,  # "gpt-4o-mini-2024-07-18",
        context_cutoff=48000,
        use_self_debug=False,
        use_knowledge=True
    )

    task = {
        "task_inst": "Using Census tract level data, which contains basic indicators from American Community Survey (ACS) data—such as poverty rate (B17020_cal), median family income (Tract_Fami), and income thresholds like State_Fa_1 or Metro_Fa_1—consider the following: Which two or three indicators would you choose to define broad economic conditions across communities? What are a few simple types of economic profiles you might expect to identify?",
        "dataset_path": "/home/zl22853/code/ai_agent/data/census_tracts_south/Census_Tracts_South.shp",
        "output_fname": "pred_results/pred.csv"
    }

    trajectory = agent.solve_task(task, out_fname="pred_programs/solution.py")
    result = agent.solve_task_ablation_direct_code(task, out_fname="pred_programs/ablation_solution_direct.py")
