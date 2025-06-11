import yaml
import os
from jinja2 import Template

class PromptManager:
    def __init__(self, prompt_dir="prompts"):
        self.prompt_dir = prompt_dir

    def load_prompt(self, agent, prompt_name):
        path = os.path.join(self.prompt_dir, agent, f"{prompt_name}.yaml")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def render_prompt(self, agent, prompt_name, variables):
        data = self.load_prompt(agent, prompt_name)
        template = Template(data['template'])
        rendered = template.render(**variables)
        return rendered

# === 示例使用 ===
if __name__ == '__main__':
    manager = PromptManager("/Users/zepingliu/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/博士学习/6-Job/ESRI/Spatial_Co_Scientist/co_scientist/prompt")

    rendered_prompt = manager.render_prompt(
        agent="generator",
        prompt_name="hypothesis",
        variables={
            "question": "How to cluster urban facilities?",
            "num_hypotheses": 3
        }
    )

    print(rendered_prompt)
