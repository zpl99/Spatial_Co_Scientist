from typing import AsyncGenerator, Generator
from typing import Optional

from .agent import Agent
from .dialog import Dialog, DialogInMemory
from .observer import AObserver, Observer
from .scratchpad import Scratchpad, ScratchpadInMemory
from .utils import u_message


class MAO:
    """Multi Agent Orchestrator.
    """

    def __init__(
            self,
            agent: Agent,
            dialog: Optional[Dialog] = None,
            scratchpad: Optional[Scratchpad] = None,
            observer: Optional[Observer] = None,
    ) -> None:
        """Initialize the Multi Agent Orchestrator.

        :param agent: The initial agent.
        :param scratchpad: Optional Scratchpad instance. If not provided, an in-memory scratchpad is used.
        :param observer: Optional Observer instance. If not provided, a no-op observer is used.
        """
        if agent is None:
            raise ValueError("Agent is required.")
        self.agent = agent
        self.dialog = dialog or DialogInMemory()
        self.scratchpad = scratchpad or ScratchpadInMemory()
        self.observer = observer or Observer.instance()
        self._terminate = False

    def terminate(self):
        """Terminate the MAO iteration.
        """
        self._terminate = True

    def __call__(
            self,
            prompt: str,
            iterations: int = 10,
    ) -> Generator:
        """Start the orchestration with a prompt.

        :param prompt: The user prompt(s) to process.
        :param iterations: The maximum number of iterations to run. Default is 10.
        """
        self.dialog += u_message(prompt)
        self._terminate = False
        iteration = 0
        self.observer.on_start()
        try:
            while not self._terminate and iteration < iterations:
                iteration += 1
                self.observer.on_iteration(iteration, self.agent.name)
                agent_response = self.agent(self.dialog, self.scratchpad, self.observer)
                if self.agent != agent_response.agent:
                    self.observer.on_handoff(self.agent, agent_response.agent)
                    self.agent = agent_response.agent
                yield agent_response
        finally:
            self.observer.on_end(iteration)


class AMAO:
    """Async Multi Agent Orchestrator.
    """

    def __init__(
            self,
            agent: Agent,
            dialog: Optional[Dialog] = None,
            scratchpad: Optional[Scratchpad] = None,
            observer: Optional[AObserver] = None,
    ) -> None:
        """Initialize the Async Multi Agent Orchestrator.

        :param agent: The initial agent.
        :param scratchpad: Optional Scratchpad instance. If not provided, an in-memory scratchpad is used.
        :param observer: Optional AObserver instance. If not provided, a no-op observer is used.
        """
        if agent is None:
            raise ValueError("Agent is required.")
        self.agent = agent
        self.dialog = dialog or DialogInMemory()
        self.scratchpad = scratchpad or ScratchpadInMemory()
        self.observer = observer or AObserver.instance()
        self._terminate = False

    def terminate(self):
        """Terminate the AMAO iteration.
        """
        self._terminate = True

    async def __call__(
            self,
            prompt: str,
            iterations: int = 10,
    ) -> AsyncGenerator:
        """Start the async orchestration with a prompt.

        :param prompt: The user prompt(s) to process.
        :param iterations: The maximum number of iterations to run. Default is 10.
        """
        self.dialog += u_message(prompt)
        self._terminate = False
        iteration = 0
        await self.observer.on_start()
        try:
            while not self._terminate and iteration < iterations:
                iteration += 1
                await self.observer.on_iteration(iteration, self.agent.name)
                agent_response = await self.agent(self.dialog, self.scratchpad, self.observer)
                if self.agent != agent_response.agent:
                    await self.observer.on_handoff(self.agent, agent_response.agent)
                    self.agent = agent_response.agent
                yield agent_response
        finally:
            await self.observer.on_end(iteration)
