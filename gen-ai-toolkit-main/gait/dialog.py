from abc import ABC, abstractmethod
from typing import List, Iterator, Optional

from .types import MessageType
from .utils import u_message


class Dialog(ABC):
    """Abstract class to implement a sequence of messages.

    A message is a dictionary with the following keys:
        role:String
        content:String
    """

    def __enter__(self):
        """Enter a context manager.

        :return: A dialog instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a context manager.

        Close open resources.
        """
        self.close()

    def __del__(self):
        """Delete a storage instance.

        Close open resources.
        """
        self.close()

    @abstractmethod
    def __iadd__(self, message: MessageType | str) -> "Dialog":
        """Add a message using the += operator.

        :param message: The chat message.
        :return: The current instance after adding the message.
        """

    @abstractmethod
    def __add__(self, message: MessageType | str) -> "Dialog":
        """Add a message using the + operator.

        :param message: The chat message.
        :return: A new instance of the dialog with the added message.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[MessageType]:
        """Allow iteration over chat messages.

        :return: An iterator over the messages.
        """

    @abstractmethod
    def clone(self) -> "Dialog":
        """Create a clone of the dialog.

        :return: A new instance of the dialog.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the storage.

        Close open resources.
        """

    @classmethod
    def instance(cls) -> "Dialog":
        """Create an instance of Dialog.

        :return: An instance of DialogInMemory.
        """
        return DialogInMemory()


class DialogSlidingWindow(Dialog):
    def __init__(self, window_size: int, messages: Optional[List[MessageType]] = None) -> None:
        """Initialize a sliding window dialog.

        :param window_size: Maximum number of messages to keep in the window.
        :param messages: Initial messages (will be trimmed to window_size if needed).
        :raises ValueError: If window_size is not positive.
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")

        self._window_size = window_size
        self._messages = list(messages or [])[-window_size:] if messages else []

    def __iadd__(self, message: MessageType | str) -> "DialogSlidingWindow":
        """Add a message to the dialog, maintaining the sliding window size.

        :param message: The message to add.
        :return: The current instance after adding the message.
        """
        msg = u_message(message) if isinstance(message, str) else message
        self._messages.append(msg)

        # Trim messages to maintain window size
        if len(self._messages) > self._window_size:
            self._messages = self._messages[-self._window_size:]

        return self

    def __add__(self, message: MessageType | str) -> "DialogSlidingWindow":
        """Create a new dialog with the added message.

        :param message: The message to add.
        :return: A new DialogSlidingWindow instance with the added message.
        """
        new_dialog = self.clone()
        new_dialog += message
        return new_dialog

    def __iter__(self) -> Iterator[MessageType]:
        """Iterate over messages in the sliding window.

        :return: An iterator over the messages.
        """
        return iter(self._messages)

    def clone(self) -> "DialogSlidingWindow":
        """Create a deep copy of the dialog.

        :return: A new DialogSlidingWindow instance with the same messages.
        """
        return DialogSlidingWindow(self._window_size, self._messages.copy())

    def close(self) -> None:
        """Close the dialog (no-op for in-memory storage)."""
        pass


class DialogInMemory(Dialog):
    def __init__(self, messages: List[MessageType] = None) -> None:
        """Initialize an in-memory dialog.

        :param messages: Initial messages.
        """
        self._messages = messages or []

    def __iadd__(self, message: MessageType | str) -> Dialog:
        """Save a message using the += operator.

        :param message: The message.
        :return: The current instance after adding the message.
        """
        match message:
            case str():
                self._messages.append(u_message(message))
            case _:
                self._messages.append(message)
        return self

    def __add__(self, message: MessageType | str) -> Dialog:
        """Save a message using the + operator.

        :param message: The message.
        :return: A new instance of the dialog with the added message.
        """
        dialog = self.clone()
        dialog += message
        return dialog

    def __iter__(self) -> Iterator[MessageType]:
        """Allow iteration over messages.

        :return: An iterator over the messages.
        """
        return iter(self._messages)

    def clone(self) -> "Dialog":
        """Create a clone of the dialog.

        :return: A new instance of the dialog.
        """
        return DialogInMemory(self._messages.copy())

    def close(self) -> None:
        """Close the storage.
        """
        return
