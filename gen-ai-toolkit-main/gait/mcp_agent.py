import json
import uuid
from typing import List, Optional

from litellm import acompletion
from litellm import experimental_mcp_client as mcp_client
from litellm.experimental_mcp_client.tools import transform_mcp_tool_to_openai_tool
from mcp import ClientSession

from .agent import AgentResponse
from .dialog import Dialog
from .mcp_server import MCPServer
from .observer import AObserver, AObserverNoop
from .scratchpad import Scratchpad, ScratchpadInMemory
from .types import InstructionType
from .utils import s_message, t_message


class MCPAgent:
    """An MCP Agent that can interact with one or more MCP servers.
    """

    def __init__(
            self,
            model: str,
            instructions: Optional[InstructionType] = None,
            mcp_servers: List[MCPServer | ClientSession] = None,
            name: str = None,
            description: str = None,
            temperature: float = 0.0,
            **kwargs,
    ):
        """Initialize the MCP agent.

        :param model: The model to use for the agent. This is required.
        :param instructions: The instructions to the agent. This is the system prompt. This can be a string or a callable.
        :param mcp_servers: A list of MCPServer instances that the agent can interact with.
        :param name: The name of the agent.
        :param temperature: The temperature to use for the model. Default is 0.0.
        :param kwargs: Additional parameters to pass to the model.
        """
        if model is None or model.strip() == "":
            raise ValueError("Model must be provided for MCPAgent.")
        self.model = model
        self.description = description or "An MCP Agent that can interact with one or more MCP servers."
        self.instructions = instructions or "You are a helpful AI agent with access to MCP (Model Context Protocol) servers and tools."
        self.mcp_servers = mcp_servers or []
        self.name = name or f"MCPAgent-{model}-{uuid.uuid4()}"
        self.temperature = temperature
        self.params = kwargs
        self.system_prompt: str = None  # To be populated after the first call.
        self.messages: List[dict] = []  # The messages that the agent has processed.
        # List of tools available to the agent from all MCP servers.
        self._tools_list = []
        self._tools_dict: dict[str, MCPServer | ClientSession] = {}  # Map tool name to MCP server.

    def __eq__(self, other) -> bool:
        """Check equality based on the model name.
        """
        if not isinstance(other, MCPAgent):
            return False
        return self.model == other.model

    async def init_tools(self) -> None:
        """Fetch tools from all MCP servers and populate the tool list.
        Note this method is there to not mix with the __init__ sync method.
        """
        for server in self.mcp_servers:
            if not isinstance(server, (MCPServer, ClientSession)):
                raise TypeError(f"Expected MCPServer or ClientSession, got {type(server)}")

            # Here this should work for both MCPServer and ClientSession.
            # TODO: Code Smell - make server list_tools() return the same type as ClientSession.list_tools()
            mcp_tools = await server.list_tools()
            if isinstance(server, ClientSession):
                mcp_tools = mcp_tools.tools
            self._tools_list.extend([
                transform_mcp_tool_to_openai_tool(tool)
                for tool
                in mcp_tools
            ])
            self._tools_dict.update({
                tool.name: server for tool in mcp_tools
            })

    async def __aenter__(self) -> "MCPAgent":
        """
        The __aenter__ approach is the cleanest and most pythonic way to handle async initialization for
        context managers
        :return: self
        """
        await self.init_tools()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def __call__(
            self,
            dialog: Dialog | str,
            scratchpad: Scratchpad = ScratchpadInMemory(),
            observer: AObserver = AObserverNoop(),
    ) -> AgentResponse:
        _dialog = dialog if isinstance(dialog, Dialog) else Dialog.instance() + str(dialog)
        agent_response = AgentResponse(agent=self)
        # Execute the instructions with scratchpad if callable.
        self.system_prompt = self.instructions(scratchpad) if callable(self.instructions) else self.instructions
        # Create the messages
        messages = [s_message(self.system_prompt)]
        messages.extend(_dialog)

        # If params has a key "tools" pull it out and create a new tools variable that is a merge between self._tools and tools
        tools = self._tools_list
        if "tools" in self.params:
            additional_tools = self.params.pop("tools")
            tools = self._tools_list + additional_tools

        params = {
            "model": self.model,
            "tools": tools,
            "temperature": self.temperature,
            "messages": messages,
            "n": 1,
            **self.params,
        }
        comp_resp = await acompletion(**params)
        for choice in comp_resp.choices:
            agent_response.finish_reason = choice.finish_reason
            choice_message = choice.message
            # Append the choice message to the dialog.
            _dialog += choice_message
            await observer.on_content(choice_message.content)
            agent_response.content = choice_message.content
            for tool_call in choice_message.tool_calls or []:
                await observer.on_function(tool_call.function.name, tool_call.function.arguments)
                arr = []
                if tool_call.function.name in self._tools_dict:
                    # Get the MCP server for the tool.
                    mcp_server = self._tools_dict[tool_call.function.name]
                    if isinstance(mcp_server, ClientSession):
                        # Directly call with ClientSession
                        call_result = await mcp_client.call_openai_tool(
                            session=mcp_server,
                            openai_tool=tool_call,
                        )
                    elif isinstance(mcp_server, MCPServer):
                        call_result = await mcp_server.call_tool(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments),
                        )
                    else:
                        raise TypeError(f"Expected MCPServer or ClientSession, got {type(mcp_server)}.")
                    if call_result.isError:
                        arr.append(f"Error calling tool {tool_call.function.name}")
                    for content in call_result.content or []:
                        # TODO: Handle more content types as needed.
                        if content.type == "text":
                            arr.append(content.text)
                else:
                    arr.append(f"Error: Tool {tool_call.function.name} not found in any MCP server.")
                content = "\n".join(arr)
                await observer.on_observation(content)
                # Append the content to the dialog.
                _dialog += t_message(
                    content,
                    tool_call.function.name,
                    tool_call.id)

        # Save the messages as a property of the agent for reference.
        self.messages = [_ for _ in _dialog]
        return agent_response
