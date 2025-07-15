import json
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any, Optional

from litellm import completion
from pydantic import BaseModel

from .dialog import Dialog
from .observer import Observer, ObserverNoop
from .scratchpad import Scratchpad, ScratchpadInMemory, __SCRATCHPAD__
from .types import InstructionType
from .utils import function_to_tool, pydantic_to_tool, t_message, s_message


@dataclass
class AgentResponse:
    """Agent response.

    :param finish_reason: The reason why the agent finishes the conversation. Like stop, tool_call, etc..
    :param content: The content of the agent response.
    :param agent: Optional agent to transfer the conversation to.
    """
    finish_reason: Optional[str] = None
    content: Optional[str] = None
    agent: Optional["Agent"] = None


@dataclass
class Agent:
    """An agent that can perform functions.

    :param name: The name of the agent.
    :param description: The description of the agent.
    :param model: The model to use for the agent. Default is 'ollama_chat/llama3.2:latest'.
    :param instructions: The instructions to the agent. This is the system prompt. This can be a string or a callable.
    :param functions: The functions that the agent can use.
    :param params: Optional extra LiteLLM parameters to pass to the model completion.
    """
    model: str = "ollama_chat/llama3.2:latest"
    name: str = None
    description: str = "You are a helpful AI agent."
    # If instructions is a callable, then the function should accept a scratchpad as an argument.
    instructions: Optional[InstructionType] = None
    functions: List[Callable | BaseModel] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    """The messages that the agent has processed.
    This is used to keep track of the conversation history.
    And will be fully populated after the first call to the agent.
    """

    def __init__(
            self,
            model: str = "ollama_chat/llama3.2:latest",
            name: str = None,
            description: str = "You are a helpful AI agent.",
            instructions: Optional[InstructionType] = None,
            functions: List[Callable] = None,
            **kwargs,
    ) -> None:
        """Initialize the agent with kwargs.

        :param model: The model to use for the agent. Default is 'ollama_chat/llama3.2:latest'.
        :param name: The name of the agent.
        :param description: The description of the agent. This is the system prompt when the agent is started.
        :param instructions: The instructions to show when the agent is started. This can be a string or a callable.
        :param functions: The functions that the agent can use.
        :param kwargs: Optional extra LLMLite parameters to pass to the model.
        """
        if model is None or model.strip() == "":
            raise ValueError("Parameter model is required.")
        self.model = model
        self.name = name
        self.description = description
        self.instructions = instructions if instructions is not None else description
        self.functions = functions if functions is not None else []
        self.params = kwargs or {}

    def __eq__(self, other: object) -> bool:
        """Override the equal operator to compare agents by name only."""
        if not isinstance(other, Agent):
            return ValueError(f"Can only compare Agent with another Agent. Got other type is {type(other)}.")
        return self.name == other.name

    def __ne__(self, other: object) -> bool:
        """Override the not equal operator to compare agents by name only."""
        return not self.__eq__(other)

    def __post_init__(self):
        """Convert the description to instructions if instructions is None.
        """
        if self.instructions is None:
            self.instructions = self.description
        if self.name is None:
            self.name = f"Agent-{self.model}"

    def _func_dict(self) -> Dict[str, Callable]:
        """Convert the functions to a dictionary with function names as keys."""
        d1 = {func.__name__: func for func in self.functions if hasattr(func, "__name__")}
        d2 = {func.__class__.__name__: func for func in self.functions if
              hasattr(func, "__class__") and hasattr(func.__class__, "__name__")}
        return d1 | d2

    def __call__(
            self,
            dialog: Dialog | str,
            scratchpad: Scratchpad = ScratchpadInMemory(),
            observer: Observer = ObserverNoop(),
    ) -> AgentResponse:
        """Process the dialog and return an agent response.

        :param dialog: An instance of Dialog or a string.
        :param scratchpad: The scratchpad to use for the agent. Default is an in-memory scratchpad.
        :param observer: The observer to use for the agent. Default is a no-op observer.
        :return: The agent response.
        """
        # Create an instance of Dialog if dialog is a string.
        _dialog = dialog if isinstance(dialog, Dialog) else Dialog.instance() + str(dialog)
        agent_response = AgentResponse(agent=self)
        # Execute the instructions with scratchpad if callable.
        content = self.instructions(scratchpad) if callable(self.instructions) else self.instructions
        # TODO: Save content into self._system_prompt and expose as a property.
        # Create the messages
        messages = [s_message(content)]
        messages.extend(_dialog)
        # Convert the functions to tools.
        tools = [function_to_tool(_) for _ in self.functions if isinstance(_, Callable)]
        tools.extend([pydantic_to_tool(_) for _ in self.functions if isinstance(_, BaseModel)])
        params = {
            "model": self.model,
            "messages": messages,
            "tools": tools or None,
            **self.params,
        }
        # Invoke the model.
        comp_resp = completion(**params)
        for choice in comp_resp.choices:
            agent_response.finish_reason = choice.finish_reason
            choice_message = choice.message
            # Append the choice message to the dialog.
            _dialog += choice_message
            observer.on_content(choice_message.content)
            agent_response.content = choice_message.content
            # func_dict = {_.__name__: _ for _ in self.functions}
            func_dict = self._func_dict()
            # Invoke the functions if any.
            for tool_call in choice_message.tool_calls or []:
                observer.on_function(tool_call.function.name, tool_call.function.arguments)
                func_name = tool_call.function.name
                if func_name in func_dict:
                    func = func_dict[func_name]
                    args = json.loads(tool_call.function.arguments)
                    # Invoke the function and get the content.
                    match func:
                        case _ if isinstance(func, type) and issubclass(func, BaseModel):
                            # Here we assume the function is a pydantic model and has a __call__ method.
                            try:
                                content = func(**args)(__SCRATCHPAD__=scratchpad)
                            except Exception as e:
                                content = "\n".join([
                                    f"Error in calling the tool '{func_name}' with {tool_call.function.arguments}.",
                                    str(e),
                                    # traceback.format_exc(),
                                ])
                        case _ if isinstance(func, Callable):
                            # If the function has a scratchpad argument, then add it to the arguments.
                            if __SCRATCHPAD__ in func.__code__.co_varnames:
                                args[__SCRATCHPAD__] = scratchpad
                            try:
                                content = func(**args)
                            except Exception as e:
                                content = "\n".join([
                                    f"Error in calling the tool '{func_name}' with {tool_call.function.arguments}.",
                                    str(e),
                                    # traceback.format_exc(),
                                ])
                        case _:
                            raise ValueError(f"Unsupported function type: {type(func)}")
                    # Update the content based on the type.
                    match content:
                        case bool() | int() | float() | str():
                            content = str(content)
                        case dict() | list():
                            content = json.dumps(content, ensure_ascii=False)
                        case Agent() as agent:
                            agent_response.agent = agent
                            content = f"Transfer to agent '{agent.name}'."
                        # Handle the case when the function returns a pydantic model.
                        case BaseModel() as model:
                            content = model.model_dump_json()
                        case _:
                            raise ValueError(f"Unsupported response type: {type(content)}")
                    observer.on_observation(content)
                    # Append the content to the dialog.
                    _dialog += t_message(
                        content,
                        func_name,
                        tool_call.id)
                else:
                    raise ValueError(f"Error: Function '{func_name}' is not found.")
            break  # Process the first choice only.

        # Update the messages in the agent response.
        self.messages = [_ for _ in _dialog]
        return agent_response
