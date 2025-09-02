from contextlib import asynccontextmanager
from typing import AsyncIterator
from dataclasses import dataclass


@dataclass
class LocalKnowledgeBase:
    """Basic in-memory knowledge base for testing."""

    data: dict[str, str]

    @classmethod
    @asynccontextmanager
    async def connect(cls) -> AsyncIterator["LocalKnowledgeBase"]:
        try:
            yield cls(data={})
        finally:
            del cls

    async def query(self, key: str) -> str:
        return self.data[key] if key in self.data else "No data found for key"

    async def update(self, key: str, value: str) -> None:
        self.data[key] = value
