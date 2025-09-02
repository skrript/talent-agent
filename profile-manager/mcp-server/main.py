import json
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context
from knowledgebase import KnowledgeBaseProtocol, LocalKnowledgeBase


@dataclass
class ServerContext:
    kb: KnowledgeBaseProtocol


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[ServerContext]:
    async with LocalKnowledgeBase.connect() as kb:
        yield ServerContext(kb=kb)


server = FastMCP(lifespan=lifespan)

if __name__ == "__main__":
    server.run("stdio")
