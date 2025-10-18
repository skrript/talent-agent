import json
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context

from graph.base import GraphDB
from graph.aws_neptune import NeptuneClient


@dataclass
class ServerContext:
    db: GraphDB


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[ServerContext]:
    async with NeptuneClient() as db:
        yield ServerContext(db=db)


server = FastMCP(lifespan=lifespan)


@server.prompt(name="profileManager", description="Manages user's work profile.")
def system_prompt() -> str:
    return """

"""


@server.tool(title="queryKnowledgeBase")
async def query_knowledge_base(ctx: ServerContext, key: str):
    pass


if __name__ == "__main__":
    server.run("stdio")
