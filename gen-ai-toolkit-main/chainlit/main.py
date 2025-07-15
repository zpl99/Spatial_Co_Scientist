import os
from textwrap import dedent

import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput
from mcp import ClientSession

from gait import Layers, MCPAgent, DialogInMemory, AMAO, AObserver


class CLObserver(AObserver):
    """ChainLit implementation of AObserver that sends messages to the UI."""

    async def on_start(self) -> None:
        """Called when GAIT LLM starts."""
        await cl.Message(content="🚀 Agent started",
                         # author="system"
                         ).send()

    async def on_end(self, iterations: int) -> None:
        """Called when GAIT LLM ends."""
        await cl.Message(content=f"✅ Agent completed after {iterations} iterations",
                         # author="system"
                         ).send()

    async def on_iteration(self, iteration: int, agent_name: str) -> None:
        """Called when an iteration starts."""
        await cl.Message(content=f"🔄 Iteration {iteration} ({agent_name}) started",
                         # author="system"
                         ).send()

    async def on_content(self, content: str) -> None:
        """Called on a content."""
        if content and content.strip():
            await cl.Message(content=f"💭 {content}",
                             # author="assistant"
                             ).send()

    async def on_function(self, func_name: str, func_args: str) -> None:
        """Called when an action/tool/function is about to be executed."""
        async with cl.Step(name=func_name) as step:
            step.input = func_args

    async def on_observation(self, observation: str) -> None:
        """Called when an observation is made."""
        if observation and observation.strip():
            await cl.Message(content=f"📋 Tool result: {observation}",
                             # author="system"
                             ).send()

    async def on_handoff(self, from_agent, to_agent) -> None:
        """Called when a handoff occurs between agents."""
        await cl.Message(content=f"🤝 Handoff from {from_agent.name} to {to_agent.name}",
                         # author="system"
                         ).send()


def layers_to_instructions(layers: Layers) -> str:
    """Convert layer information to system prompt instructions.
    """
    lines = []
    for layer in layers.layers:
        lines.append(f"Layer: {layer.name}")
        lines.append(f"Alias: {layer.alias}")
        lines.append(f"Spatial Type: {layer.stype}")
        if layer.hints:
            lines.append("Hints:")
            for hint in layer.hints:
                lines.append(f"  - {hint}")

        if layer.columns:
            lines.append("Columns:")
            for column in layer.columns:
                lines.append(f"  - Name: {column.name}")
                lines.append(f"    Type: {column.dtype}")
                if column.hints:
                    lines.append("    Hints:")
                    for hint in column.hints:
                        lines.append(f"      - {hint}")
                if column.values:
                    sample_values = column.values[:3]
                    lines.append(f"    Sample Values: {sample_values}")
        lines.append("")  # Add empty line between layers

    return "\n".join(lines)


@cl.cache
def get_instructions(layers_json: str) -> str:
    template = dedent("""
        You are an expert GIS analyst. You will be provided with a user prompt and a description of GIS layers.
        The following are the GIS layers with their shape types and field information:

        {layers}
        
        Instructions:
        - Understand the user's prompt and identify relevant GIS layers and fields.
        - Make sure that sample questions to not contain aggregate functions like summation, average, min, max, count, etc.
        - Make sure to NOT add any markdown, HTML, or code blocks in your response.
        
        Now, answer the user prompt based on the above GIS layers.
        """).strip()
    layers = layers_to_instructions(Layers.load(layers_json))
    return template.format(layers=layers)


@cl.on_chat_start
async def on_chat_start():
    # Read the layers_json value from the config file
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=[
                    f"azure/{os.environ['AZURE_API_DEPLOYMENT']}",
                    "claude-3-7-sonnet-20250219",
                ],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.0,
                min=0.0,
                max=2.0,
                step=0.1,
            ),
            TextInput(
                id="layers_path",
                label="Layers JSON Path",
                initial=os.path.expanduser(os.environ.get('LAYERS_JSON', "MissingInDotEnv")),
            )
        ]).send()
    cl.user_session.set("settings", settings)
    cl.user_session.set("dialog", DialogInMemory())
    cl.user_session.set("client_sessions", {})


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established.
    This handler is required for MCP to work
    """
    client_sessions = cl.user_session.get("client_sessions")
    client_sessions[connection.name] = session
    cl.user_session.set("client_sessions", client_sessions)


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated.
    """
    client_sessions = cl.user_session.get("client_sessions")
    if name in client_sessions:
        del client_sessions[name]
        cl.user_session.set("client_sessions", client_sessions)


@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings", {})
    instructions = get_instructions(settings["layers_path"])
    client_sessions = cl.user_session.get("client_sessions", {})
    async with MCPAgent(model=settings["model"],
                        instructions=instructions,
                        mcp_servers=list(client_sessions.values()),
                        temperature=settings["temperature"],
                        ) as mcp_agent:
        mao = AMAO(
            agent=mcp_agent,
            dialog=cl.user_session.get("dialog"),
            observer=CLObserver(),
        )
        async for _ in mao(prompt=message.content):
            # Terminate the loop if there is a content.
            if _.content:
                mao.terminate()
