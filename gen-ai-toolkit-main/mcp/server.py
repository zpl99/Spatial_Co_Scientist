import random
from enum import Enum

import typer
# from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing_extensions import Annotated


class TemperatureResponse(BaseModel):
    location: str
    temperature: float
    unit: str = "Celsius"


# load_dotenv()

mcp = FastMCP("G.AI.T")


@mcp.tool()
def get_temperature(location: str) -> TemperatureResponse:
    """Get the temperature in a city

    Args:
        location: The city to get the temperature in.

    Returns:    
        A dictionary containing the city, temperature, and unit
    """
    temperature = random.uniform(-5, 40)
    return TemperatureResponse(
        location=location,
        temperature=round(temperature, 2),
        unit="Celsius",
    )


class Transport(str, Enum):
    stdio = "stdio"
    sse = "sse"


def main(
        transport: Annotated[Transport, typer.Argument(
            help="MCP transport mode."
        )] = Transport.sse,
):
    mcp.run(
        transport=transport,
    )


if __name__ == "__main__":
    typer.run(main)
