from textwrap import dedent

from litellm import completion

from .fel_base import FEL0
from .fel_observer import FELObserverABC, FELObserverNoop
from ..flow import Node
from ..scratchpad import Scratchpad
from ..utils import u_message, s_message, a_message

# litellm._turn_on_debug()

FELLLM0_SYSTEM = dedent(
    """
You are a highly skilled AI assistant designed to analyze user queries and accurately determine the appropriate layers as a JSON object.

The user will ask questions related to one or more of the following geospatial layers.
Use the provided layer and field definitions to identify the correct layer and map the user’s request to a valid layers in JSON format.
Ensure your response is concise and emits only the JSON result.
Accuracy is CRITICAL.

{attributes}

Instructions:
1. Understand the user query and identify the relevant geospatial layer(s) and attribute(s) based on the schema provided.
2. Construct the correct layers as a JSON document containing the layer names.
3. Output only the JSON result, with no additional formatting, explanations, or tags.
4. Ensure your answer is 100% accurate, as a reward of $1,000,000 is contingent upon correctness.
"""
).strip()


class FELLLM0(Node):
    def __init__(self, observer: FELObserverABC = None) -> None:
        super().__init__()
        self.observer = observer or FELObserverNoop()

    def exec(self, sp: Scratchpad) -> str:

        def _logger_fn(model_call_dict):
            self.observer.on_message(str(model_call_dict))

        attributes = []
        for l in sp["layers"]:
            attributes.append(f"Fields for layer {l.alias}:")
            for c in l.columns_no_geom:
                attributes.append(f"- {c.alias}")
            attributes.append("\n")

        messages = [
            s_message(FELLLM0_SYSTEM.format(attributes=("\n".join(attributes)))),
        ]
        for fel_line in sp["fel0"]:
            self.observer.on_fel_line(fel_line)
            messages.append(u_message(fel_line.line))
            messages.append(a_message(fel_line.fel.model_dump_json()))

        messages.append(u_message(sp["prompt"]))
        self.observer.on_message("FELLLM0::completion...")
        response = completion(
            model=sp["model"],
            response_format=FEL0,
            temperature=0.0,
            messages=messages,
            # logger_fn=_logger_fn,
        )
        choice = response.choices[0]
        self.observer.on_message(str(choice))
        fel0 = FEL0.model_validate_json(choice.message.content)
        self.observer.on_fel0(fel0)
        sp["llm0"] = fel0
        match bool(fel0.layer1), bool(fel0.layer2):
            case (True, False):
                return "1"
            case (True, True):
                return "2"
            case _:
                return Node.DEFAULT
