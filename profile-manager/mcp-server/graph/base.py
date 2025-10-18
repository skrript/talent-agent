from typing import Self
from abc import ABC, abstractmethod


class GraphDB(ABC):
    @abstractmethod
    async def _connect(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _disconnect(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def __aenter__(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(
        self,
    ) -> None:
        raise NotImplementedError
