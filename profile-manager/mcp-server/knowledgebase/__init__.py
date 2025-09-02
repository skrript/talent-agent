"""Knowledge base implementations."""

from typing import Protocol, AsyncIterator, Self

from .local_kb import LocalKnowledgeBase


class KnowledgeBaseProtocol(Protocol):
    @classmethod
    async def connect(cls) -> AsyncIterator[Self]:
        yield cls()

    async def update(self, key: str, value: str):
        pass

    async def query(self, key: str) -> str:
        return ""


__all__ = ["KnowledgeBaseProtocol", "LocalKnowledgeBase"]
