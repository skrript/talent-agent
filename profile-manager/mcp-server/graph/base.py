from typing import Self, Any, Callable
from abc import ABC, abstractmethod


class GraphDB(ABC):
    """Base class for graph database connections."""

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


class GremlinQueryInterface(ABC):
    """Interface for databases that support Gremlin query language."""

    @abstractmethod
    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Gremlin query string."""
        raise NotImplementedError

    @abstractmethod
    async def execute_traversal(
        self, traversal_lambda: Callable[[Any], Any]
    ) -> list[dict[str, Any]]:
        """Execute a Gremlin traversal using the graph traversal source."""
        raise NotImplementedError


class GraphOperationsInterface(ABC):
    """Common interface for graph operations that work across query languages."""

    @abstractmethod
    async def add_vertex(
        self, label: str, properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Add a vertex to the graph."""
        raise NotImplementedError

    @abstractmethod
    async def add_edge(
        self,
        from_vertex_id: str,
        to_vertex_id: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add an edge between two vertices."""
        raise NotImplementedError

    @abstractmethod
    async def get_vertex(self, vertex_id: str) -> dict[str, Any] | None:
        """Get a vertex by ID."""
        raise NotImplementedError

    @abstractmethod
    async def get_schema(self) -> dict[str, Any]:
        """Get complete graph schema."""
        raise NotImplementedError

    @abstractmethod
    async def get_graph_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the graph."""
        raise NotImplementedError
