import abc
import logging
import random
import time
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from typing import Optional, Dict, Any, List

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp import Tool
from mcp.client.streamable_http import GetSessionIdCallback
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult


class MCPServer(abc.ABC):
    """Interface to a Model Context Protocol server."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the server."""
        pass

    @abc.abstractmethod
    async def connect(self):
        """Connect to the server. The server is expected to remain connected until `close()` is called."""
        pass

    @abc.abstractmethod
    async def close(self):
        """Close the connection to the server, and release any resources."""
        pass

    @abc.abstractmethod
    async def list_tools(self) -> list[Tool]:
        """List the tools available on the server."""
        pass

    @abc.abstractmethod
    async def call_tool(
            self,
            tool_name: str,
            tool_args: dict[str, Any] | None
    ) -> CallToolResult:
        """Call a tool on the server.

        :param tool_name: The name of the tool to call.
        :param tool_args: The arguments to pass to the tool
        """
        pass


class MCPServerBase(MCPServer, abc.ABC):
    """Abstract base class for a Model Context Protocol server."""

    def __init__(self, session_timeout_seconds: float | None) -> None:
        """Initialize the server with optional parameters.

        :param session_timeout_seconds: The session timeout in seconds. If None, no timeout is set.
        """
        self.session_timeout_seconds = session_timeout_seconds
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[Tool] | None = None
        self.logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def create_streams(
            self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        """Create the read/write streams for the server."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()
        return False

    async def connect(self):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Connecting to {self.name}.")
        streams = await self.exit_stack.enter_async_context(self.create_streams())
        read, write, *_ = streams
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                read,
                write,
                (
                    timedelta(seconds=self.session_timeout_seconds)
                    if self.session_timeout_seconds
                    else None
                ),
            )
        )
        await self.session.initialize()

    async def list_tools(self) -> list[Tool]:
        if self.tools is None:
            if not self.session:
                raise RuntimeError(f"{self.name} is not connected!")
            resp = await self.session.list_tools()
            self.tools = resp.tools
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Available tools: {[tool.name for tool in self.tools]}")
        return self.tools

    async def call_tool(
            self,
            tool_name: str,
            tool_args: dict[str, Any] | None,
    ) -> CallToolResult:
        if not self.session:
            raise RuntimeError(f"{self.name} is not connected!")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Calling tool {tool_name} with args {tool_args}.")
        return await self.session.call_tool(tool_name, tool_args or {})

    async def close(self):
        if self.exit_stack and self.session:
            try:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Closing {self.name}.")
                await self.exit_stack.aclose()
            except Exception as e:
                self.logger.error(f"Error closing {self.name}: {e}")
            finally:
                self.session = None


class MCPServerSSE(MCPServerBase):
    """MCP server that communicates over Server-Sent Events (SSE)."""

    def __init__(
            self,
            url: str,
            name: str | None = None,
            session_timeout_seconds: float | None = None,
            client_timeout_seconds: float = 5,
            read_timeout_seconds: float = 60 * 5,
            headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the server with the given URL and optional parameters.

        :param url: The URL of the SSE endpoint.
        :param name: The name of the server. If None, a default name is generated from the URL.
        :param session_timeout_seconds: The session timeout in seconds. If None, no timeout is set.
        :param client_timeout_seconds: The client timeout in seconds for establishing the connection.
        :param read_timeout_seconds: The read timeout in seconds for reading from the stream.
        :param headers: Optional headers to include in the SSE request.
        """
        super().__init__(session_timeout_seconds)
        if url is None or url.strip() == "":
            raise ValueError("URL must be a non-empty string.")
        self.url = url
        self.client_timeout_seconds = client_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.headers = headers
        self._name = name or f"sse: {url}"

    @property
    def name(self) -> str:
        """The name of the server."""
        return self._name

    def create_streams(
            self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        from mcp.client.sse import sse_client

        return sse_client(
            self.url,
            self.headers,
            self.client_timeout_seconds,
            self.read_timeout_seconds,
        )


class MCPServerStdio(MCPServerBase):
    """MCP server that communicates over standard input/output (stdio)."""

    def __init__(
            self,
            parameters: StdioServerParameters | None = None,
            name: str | None = None,
            session_timeout_seconds: float | None = None,
            command: str | None = None,
            args: list[str] | None = None,
    ) -> None:
        """Initialize the server with the given command and optional parameters.

        :param parameters: The parameters for the stdio server, including the command to run.
        :param name: The name of the server. If None, a default name is generated from the command.
        :param session_timeout_seconds: The session timeout in seconds. If None, no timeout is set.
        :param command: The command to run the stdio server. If None, must be provided in parameters.
        :param args: The arguments to pass to the command. If None, must be provided in parameters.
        """
        super().__init__(session_timeout_seconds)
        if parameters is None and command is None:
            raise ValueError("parameters or command must be provided.")
        if parameters is None and command is not None:
            parameters = StdioServerParameters(command=command, args=args or [], env=None)
        if parameters.command is None or parameters.command.strip() == "":
            raise ValueError("parameters.command must be a non-empty string.")
        self.parameters = parameters
        self._name = name or f"stdio: {parameters.command}"

    @property
    def name(self) -> str:
        """The name of the server."""
        return self._name

    def create_streams(
            self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        from mcp.client.stdio import stdio_client

        return stdio_client(self.parameters)


class MCPServerStreamableHTTP(MCPServerBase):
    """MCP server that communicates over a streamable HTTP connection.

    :param url: The URL of the streamable HTTP endpoint.
    :param name: The name of the server. If None, a default name is generated from the URL.
    :param session_timeout_seconds: The session timeout in seconds. If None, no timeout is set.
    :param client_timeout_seconds: The client timeout in seconds for establishing the connection.
    :param read_timeout_seconds: The read timeout in seconds for reading from the stream.
    :param headers: Optional headers to include in the HTTP request.
    """

    def __init__(
            self,
            url: str,
            name: str | None = None,
            session_timeout_seconds: float | None = None,
            client_timeout_seconds: float = 5,
            read_timeout_seconds: float = 60 * 5,
            headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the server with the given URL and optional parameters."""
        super().__init__(session_timeout_seconds)
        if url is None or url.strip() == "":
            raise ValueError("URL must be a non-empty string.")
        self.url = url
        self.client_timeout_seconds = client_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.headers = headers
        self._name = name or f"http: {url}"

    @property
    def name(self) -> str:
        """The name of the server."""
        return self._name

    def create_streams(
            self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        from mcp.client.streamable_http import streamablehttp_client

        return streamablehttp_client(
            self.url,
            self.headers,
            self.client_timeout_seconds,
            self.read_timeout_seconds,
        )


class MCPServerFake(MCPServer):
    """A fake implementation of MCPServer for testing purposes.

    This server provides a set of sample tools that can be used for testing
    without requiring a real MCP server connection.
    """

    def __init__(self, name: str = "Fake Server"):
        """Initialize the fake server.

        Args:
            name: Optional name for the server. Defaults to "Fake Server".
        """
        self._name = name
        self._connected = False
        self._tools = [
            Tool(
                name="get_time",
                description="Get the current server time in ISO format",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="generate_number",
                description="Generate a random number within a range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min": {
                            "type": "integer",
                            "description": "Minimum value (inclusive)",
                        },
                        "max": {
                            "type": "integer",
                            "description": "Maximum value (inclusive)",
                        },
                    },
                    "required": ["min", "max"],
                },
            ),
            Tool(
                name="echo",
                description="Echo back the input",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo back",
                        },
                    },
                    "required": ["message"],
                },
            ),
        ]

    @property
    def name(self) -> str:
        """The name of the server."""
        return self._name

    async def connect(self):
        """Connect to the fake server."""
        if not self._connected:
            self._connected = True

    async def close(self):
        """Close the connection to the fake server."""
        self._connected = False

    async def list_tools(self) -> List[Tool]:
        """List the tools available on the fake server."""
        if not self._connected:
            raise RuntimeError(f"{self._name} is not connected!")
        return self._tools

    async def call_tool(
            self,
            tool_name: str,
            tool_args: Optional[Dict[str, Any]] = None
    ) -> CallToolResult:
        """Call a tool on the fake server.

        Args:
            tool_name: The name of the tool to call.
            tool_args: The arguments to pass to the tool.

        Returns:
            CallToolResult: The result of the tool call.

        Raises:
            ValueError: If the tool name is not recognized.
        """
        if not self._connected:
            raise RuntimeError(f"{self._name} is not connected!")

        tool_args = tool_args or {}

        if tool_name == "get_time":
            return CallToolResult(
                content={"time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            )

        elif tool_name == "generate_number":
            min_val = tool_args.get("min", 0)
            max_val = tool_args.get("max", 100)
            return CallToolResult(content={"number": random.randint(min_val, max_val)})

        elif tool_name == "echo":
            return CallToolResult(content={"message": tool_args.get("message", "")})
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

# if __name__ == "__main__":
#     import asyncio
#     from rich.pretty import pprint
#
#
#     async def main():
#         url = "http://localhost:8000/sse"
#         async with MCPServerSSE(url) as server:
#             tools = await server.list_tools()
#             pprint(tools, expand_all=True)
#             if tools:
#                 result = await server.call_tool(
#                     "get_temperature", dict(location="New York")
#                 )
#                 pprint(result, expand_all=True)
#
#
#     asyncio.run(main())
