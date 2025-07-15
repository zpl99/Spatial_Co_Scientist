import logging
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


class Observer(ABC):
    @abstractmethod
    def on_start(self) -> None:
        """Called when GAIT LLM starts.
        """

    @abstractmethod
    def on_end(self, iterations: int) -> None:
        """Called when GAIT LLM ends.

        :param iterations: The number of iterations.
        """

    @abstractmethod
    def on_iteration(self, iteration: int, agent_name: str) -> None:
        """Called when an iteration starts.

        :param iteration: The iteration number.
        :param agent_name: The name of the active agent.
        """

    @abstractmethod
    def on_content(self, content: str) -> None:
        """Called on a content.

        :param content: The content.
        """

    @abstractmethod
    def on_function(self, func_name: str, func_args: str) -> None:
        """Called when an action/tool/function is about to be executed.

        :param func_name: The function name.
        :param func_args: The function arguments.
        """

    @abstractmethod
    def on_observation(self, observation: str) -> None:
        """Called when an observation is made, which is the return of a action call.

        :param observation: The observation.
        """

    @abstractmethod
    def on_handoff(self, from_agent: "Agent", to_agent: "Agent") -> None:
        """Called when a handoff occurs between agents.

        :param from_agent: The agent handing off the conversation.
        :param to_agent: The agent receiving the conversation.
        """

    @classmethod
    def instance(cls) -> "Observer":
        """Create an instance of an Observer.

        :return: An instance of ObserverNoop.
        """
        return ObserverNoop()


class ObserverNoop(Observer):
    """No-op GAIT observer.
    """

    def on_handoff(self, from_agent: "Agent", to_agent: "Agent") -> None:
        return

    def on_start(self) -> None:
        return

    def on_end(self, iterations: int) -> None:
        return

    def on_iteration(self, iteration: int, agent_name: str) -> None:
        return

    def on_content(self, content: str) -> None:
        return

    def on_function(self, func_name: str, func_args: str) -> None:
        return

    def on_observation(self, observation: str) -> None:
        return


class ObserverLogging(Observer):
    """Logging GAIT observer.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def on_handoff(self, from_agent: "Agent", to_agent: "Agent") -> None:
        """Called when a handoff occurs between agents.
        """
        self.logger.info(f"Handoff from {from_agent.name} to {to_agent.name}.")

    def on_start(self) -> None:
        """Called when GAIT starts.
        """
        self.logger.info("GAIT started.")

    def on_end(self, iterations: int) -> None:
        """Called when GAIT ends.

        :param iterations: The number of iterations.
        """
        self.logger.info(f"GAIT ended after {iterations} iterations.")

    def on_iteration(self, iteration: int, agent_name: str) -> None:
        """Called when an iteration starts.

        :param iteration: The iteration number.
        :param agent_name: The name of the agent.
        """
        self.logger.info(f"Iteration {iteration} ({agent_name}) started.")

    def on_content(self, content: str) -> None:
        """Called on a content.

        :param content: The content.
        """
        self.logger.info(f"Content: {content}")

    def on_function(self, func_name: str, func_args: str) -> None:
        """Called when an action is executed.

        :param func_name: The action.
        :param func_args: The action arguments.
        """
        self.logger.info(f"Function: {func_name}({func_args})")

    def on_observation(self, observation: str) -> None:
        """Called when an observation is made.

        :param observation: The observation.
        """
        self.logger.info(f"Observation: {observation}")


class ObserverLoguru(Observer):
    """Loguru GAIT observer.
    """

    def __init__(self, level: str = "INFO") -> None:
        try:
            import loguru
        except ImportError:
            raise ImportError("Please run `pip install loguru` to use ObserverLoguru.")
        self.logger = loguru.logger
        self.logger.remove()

        self.logger.add(
            sys.stderr,
            format="{time:HH:mm:ss} | {level} | {message}",
            filter="gait",
            level=level,
        )

    def on_handoff(self, from_agent: "Agent", to_agent: "Agent") -> None:
        """Called when a handoff occurs between agents."""
        self.logger.info(f"Handoff from {from_agent.name} to {to_agent.name}.")

    def on_start(self) -> None:
        """Called when GAIT starts.
        """
        self.logger.info("GAIT started.")

    def on_end(self, iterations: int) -> None:
        """Called when GAIT ends.

        :param iterations: The number of iterations.
        """
        self.logger.info(f"GAIT ended after {iterations} iterations.")

    def on_iteration(self, iteration: int, agent_name: str) -> None:
        """Called when an iteration starts.

        :param iteration: The iteration number.
        :param agent_name: The name of the agent.
        """
        self.logger.info(f"Iteration {iteration} ({agent_name}) started.")

    def on_content(self, content: str) -> None:
        """Called on a content.

        :param content: The content.
        """
        self.logger.info(f"Content: {content}")

    def on_function(self, func_name: str, func_args: str) -> None:
        """Called when an action is executed.

        :param func_name: The action.
        :param func_args: The action arguments.
        """
        self.logger.info(f"Function: {func_name}({func_args})")

    def on_observation(self, observation: str) -> None:
        """Called when an observation is made.

        :param observation: The observation.
        """
        self.logger.info(f"Observation: {observation}")


class AObserver(ABC):
    """Async version of Observer abstract base class."""

    @abstractmethod
    async def on_start(self) -> None:
        """Called when GAIT LLM starts.
        """

    @abstractmethod
    async def on_end(self, iterations: int) -> None:
        """Called when GAIT LLM ends.

        :param iterations: The number of iterations.
        """

    @abstractmethod
    async def on_iteration(self, iteration: int, agent_name: str) -> None:
        """Called when an iteration starts.

        :param iteration: The iteration number.
        :param agent_name: The name of the active agent.
        """

    @abstractmethod
    async def on_content(self, content: str) -> None:
        """Called on a content.

        :param content: The content.
        """

    @abstractmethod
    async def on_function(self, func_name: str, func_args: str) -> None:
        """Called when an action/tool/function is about to be executed.

        :param func_name: The function name.
        :param func_args: The function arguments.
        """

    @abstractmethod
    async def on_observation(self, observation: str) -> None:
        """Called when an observation is made, which is the return of a action call.

        :param observation: The observation.
        """

    @abstractmethod
    async def on_handoff(self, from_agent: "Agent", to_agent: "Agent") -> None:
        """Called when a handoff occurs between agents.

        :param from_agent: The agent handing off the conversation.
        :param to_agent: The agent receiving the conversation.
        """

    @classmethod
    def instance(cls) -> "AObserver":
        """Create an instance of an AObserver.

        :return: An instance of AObserverNoop.
        """
        return AObserverNoop()


class AObserverNoop(AObserver):
    """No-op async GAIT observer.
    """

    async def on_handoff(self, from_agent: "Agent", to_agent: "Agent") -> None:
        return

    async def on_start(self) -> None:
        return

    async def on_end(self, iterations: int) -> None:
        return

    async def on_iteration(self, iteration: int, agent_name: str) -> None:
        return

    async def on_content(self, content: str) -> None:
        return

    async def on_function(self, func_name: str, func_args: str) -> None:
        return

    async def on_observation(self, observation: str) -> None:
        return


class AObserverLoguru(AObserver):
    """Async Loguru GAIT observer.
    """

    def __init__(self, level: str = "INFO") -> None:
        try:
            import loguru
        except ImportError:
            raise ImportError("Please run `pip install loguru` to use AObserverLoguru.")
        self.logger = loguru.logger
        self.logger.remove()

        self.logger.add(
            sys.stderr,
            format="{time:HH:mm:ss} | {level} | {message}",
            filter="gait",
            level=level,
        )

    async def on_handoff(self, from_agent: "Agent", to_agent: "Agent") -> None:
        """Called when a handoff occurs between agents."""
        self.logger.info(f"Handoff from {from_agent.name} to {to_agent.name}.")

    async def on_start(self) -> None:
        """Called when GAIT starts.
        """
        self.logger.info("GAIT started.")

    async def on_end(self, iterations: int) -> None:
        """Called when GAIT ends.

        :param iterations: The number of iterations.
        """
        self.logger.info(f"GAIT ended after {iterations} iterations.")

    async def on_iteration(self, iteration: int, agent_name: str) -> None:
        """Called when an iteration starts.

        :param iteration: The iteration number.
        :param agent_name: The name of the agent.
        """
        self.logger.info(f"Iteration {iteration} ({agent_name}) started.")

    async def on_content(self, content: str) -> None:
        """Called on a content.

        :param content: The content.
        """
        self.logger.info(f"Content: {content}")

    async def on_function(self, func_name: str, func_args: str) -> None:
        """Called when an action is executed.

        :param func_name: The action.
        :param func_args: The action arguments.
        """
        self.logger.info(f"Function: {func_name}({func_args})")

    async def on_observation(self, observation: str) -> None:
        """Called when an observation is made.

        :param observation: The observation.
        """
        self.logger.info(f"Observation: {observation}")
