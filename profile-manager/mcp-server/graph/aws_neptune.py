import asyncio
import logging
from typing import Self, Any, Callable
from dataclasses import dataclass

from aiogremlin import DriverRemoteConnection, Cluster
from aiogremlin.driver.client import Client
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversalSource
from gremlin_python.process.graph_traversal import __

from .base import GraphDB, GremlinQueryInterface, GraphOperationsInterface


logger = logging.getLogger(__name__)


@dataclass
class NeptuneClient(GraphDB, GremlinQueryInterface, GraphOperationsInterface):
    """AWS Neptune implementation supporting Gremlin queries and graph operations."""

    endpoint: str
    port: int = 8182
    traversal_source: str = "g"
    timeout: int = 30
    max_workers: int = 4

    _connected: bool = False
    _cluster: Cluster | None = None
    _client: Client | None = None
    _g: GraphTraversalSource | None = None
    _connection: DriverRemoteConnection | None = None

    @property
    def conn_str(
        self,
    ) -> str:
        return f"wss://{self.endpoint}:{self.port}/gremlin"

    async def __aenter__(
        self,
    ) -> Self:
        await self._connect()
        return self

    async def __aexit__(
        self,
    ):
        await self._disconnect()

    async def _connect(
        self,
    ):
        """
        Establish connection to AWS Neptune.

        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            logger.warning("Client already connected")
            return

        try:
            logger.info(f"Connecting to Neptune at {self.endpoint}:{self.port}")

            self._cluster = await Cluster.open(
                url=self.conn_str,
                aliases={self.traversal_source: self.traversal_source},
                pool_size=self.max_workers,
                response_timeout=self.timeout,
                loop=asyncio.get_event_loop(),
            )

            self._client = await self._cluster.connect()

            self._connection = await DriverRemoteConnection.using(
                self._cluster, aliases={self.traversal_source: self.traversal_source}
            )

            self._g = traversal().with_remote(self._connection)

            await self._g.inject(1).next()

            self._connected = True
            logger.info("Successfully connected to Neptune")

        except Exception as e:
            logger.error(f"Failed to connect to Neptune: {e}")
            self._connected = False
            self._cluster, self._client, self._connection, self._g = (
                None,
                None,
                None,
                None,
            )
            raise ConnectionError(f"Could not connect to Neptune: {e}") from e

    async def _disconnect(
        self,
    ):
        try:
            logger.info("Closing Neptune connection")
            if self._client:
                await self._client.close()
            if self._cluster:
                await self._cluster.close()
            if self._connection:
                await self._connection.close()
            self._connected = False
            self._cluster, self._client, self._connection, self._g = (
                None,
                None,
                None,
                None,
            )
            logger.info("Neptune connection closed")

        except Exception as e:
            logger.error(f"Error closing Neptune connection: {e}")
            raise

    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a Gremlin query string.

        Args:
            query: Gremlin query string
            params: Optional query parameters (bindings)

        Returns:
            List of result dictionaries

        Raises:
            ConnectionError: If not connected
            Exception: If query execution fails
        """
        if not self._connected or not self._connection:
            raise ConnectionError(
                "Client not connected. Use 'async with' or call connect() first"
            )

        try:
            logger.debug(f"Executing query: {query[:100]}...")
            result_set = await self._client.submit(query, bindings=params or {})

            # Convert result stream to list
            result_list = await result_set.all()

            logger.debug(f"Query returned {len(result_list)} results")

            return self._normalize_results(result_list)

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    async def execute_traversal(
        self, traversal_lambda: Callable[[Any], Any]
    ) -> list[dict[str, Any]]:
        """
        Execute a Gremlin traversal using the graph traversal source.

        Args:
            traversal_lambda: Function that takes 'g' and returns a traversal
                            Example: lambda g: g.V().hasLabel('User').limit(10)

        Returns:
            List of result dictionaries

        Raises:
            ConnectionError: If not connected
        """
        if not self._connected or not self._g:
            raise ConnectionError(
                "Client not connected. Use 'async with' or call connect() first"
            )

        try:
            # Execute the traversal
            t = traversal_lambda(self._g)
            results = await t.to_list()

            return self._normalize_results(results)

        except Exception as e:
            logger.error(f"Error executing traversal: {e}")
            raise

    def _normalize_results(self, results: list[Any]) -> list[dict[str, Any]]:
        """
        Normalize query results to list of dictionaries.

        Handles different result types from Gremlin queries.
        """
        normalized = []

        for result in results:
            if isinstance(result, dict):
                normalized.append(result)
            elif hasattr(result, "__dict__"):
                # Convert objects to dicts
                normalized.append(vars(result))
            else:
                # Wrap primitives
                normalized.append({"value": result})

        return normalized

    async def add_vertex(
        self, label: str, properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Add a vertex to the graph.

        Args:
            label: Vertex label
            properties: Optional vertex properties

        Returns:
            Created vertex data
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        props = properties or {}

        # Build traversal
        t = self._g.add_v(label)
        for key, value in props.items():
            t = t.property(key, value)

        result = await t.element_map().next()
        return result

    async def add_edge(
        self,
        from_vertex_id: str,
        to_vertex_id: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add an edge between two vertices.

        Args:
            from_vertex_id: Source vertex ID
            to_vertex_id: Target vertex ID
            label: Edge label
            properties: Optional edge properties

        Returns:
            Created edge data
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        props = properties or {}

        # Build traversal
        t = self._g.V(from_vertex_id).add_e(label).to(self._g.V(to_vertex_id))
        for key, value in props.items():
            t = t.property(key, value)

        result = await t.element_map().next()
        return result

    async def get_vertex(self, vertex_id: str) -> dict[str, Any] | None:
        """
        Get a vertex by ID.

        Args:
            vertex_id: Vertex ID

        Returns:
            Vertex data or None if not found
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        try:
            result = await self._g.V(vertex_id).element_map().next()
            return result
        except StopAsyncIteration:
            return None

    async def get_vertex_labels(self) -> list[str]:
        """
        Get all unique vertex labels in the graph.

        Returns:
            List of vertex label strings
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        results = await self._g.V().label().dedup().to_list()
        return results

    async def get_edge_labels(self) -> list[str]:
        """
        Get all unique edge labels in the graph.

        Returns:
            List of edge label strings
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        results = await self._g.E().label().dedup().to_list()
        return results

    async def get_vertex_properties(
        self, label: str | None = None
    ) -> dict[str, list[str]]:
        """
        Get all property keys for vertices, optionally filtered by label.

        Args:
            label: Optional vertex label to filter by

        Returns:
            Dictionary mapping vertex labels to their property keys
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        if label:
            props = (
                await self._g.V().has_label(label).properties().key().dedup().to_list()
            )
            return {label: props}
        else:
            # Get properties for all labels
            labels = await self.get_vertex_labels()
            schema = {}
            for lbl in labels:
                props = (
                    await self._g.V()
                    .has_label(lbl)
                    .properties()
                    .key()
                    .dedup()
                    .to_list()
                )
                schema[lbl] = props
            return schema

    async def get_edge_properties(
        self, label: str | None = None
    ) -> dict[str, list[str]]:
        """
        Get all property keys for edges, optionally filtered by label.

        Args:
            label: Optional edge label to filter by

        Returns:
            Dictionary mapping edge labels to their property keys
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        if label:
            props = (
                await self._g.E().has_label(label).properties().key().dedup().to_list()
            )
            return {label: props}
        else:
            labels = await self.get_edge_labels()
            schema = {}
            for lbl in labels:
                props = (
                    await self._g.E()
                    .has_label(lbl)
                    .properties()
                    .key()
                    .dedup()
                    .to_list()
                )
                schema[lbl] = props
            return schema

    async def get_schema(self) -> dict[str, Any]:
        """
        Get complete graph schema including vertices, edges, and their properties.

        Returns:
            Dictionary with complete schema information
        """
        vertex_labels = await self.get_vertex_labels()
        edge_labels = await self.get_edge_labels()

        vertex_properties = await self.get_vertex_properties()
        edge_properties = await self.get_edge_properties()

        # Get edge connections (which vertex labels connect to which)
        edge_connections = {}
        for edge_label in edge_labels:
            results = await (
                self._g.E()
                .has_label(edge_label)
                .project("from", "to")
                .by(__.out_v().label())
                .by(__.in_v().label())
                .dedup()
                .to_list()
            )
            edge_connections[edge_label] = results

        return {
            "vertices": {"labels": vertex_labels, "properties": vertex_properties},
            "edges": {
                "labels": edge_labels,
                "properties": edge_properties,
                "connections": edge_connections,
            },
        }

    async def get_vertex_count(self, label: str | None = None) -> int:
        """
        Get count of vertices, optionally filtered by label.

        Args:
            label: Optional vertex label to filter by

        Returns:
            Count of vertices
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        if label:
            count = await self._g.V().has_label(label).count().next()
        else:
            count = await self._g.V().count().next()

        return count

    async def get_edge_count(self, label: str | None = None) -> int:
        """
        Get count of edges, optionally filtered by label.

        Args:
            label: Optional edge label to filter by

        Returns:
            Count of edges
        """
        if not self._connected or not self._g:
            raise ConnectionError("Client not connected")

        if label:
            count = await self._g.E().has_label(label).count().next()
        else:
            count = await self._g.E().count().next()

        return count

    async def get_graph_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about the graph.

        Returns:
            Dictionary with graph statistics
        """
        vertex_labels = await self.get_vertex_labels()
        edge_labels = await self.get_edge_labels()

        vertex_counts = {}
        for label in vertex_labels:
            vertex_counts[label] = await self.get_vertex_count(label)

        edge_counts = {}
        for label in edge_labels:
            edge_counts[label] = await self.get_edge_count(label)

        return {
            "total_vertices": sum(vertex_counts.values()),
            "total_edges": sum(edge_counts.values()),
            "vertex_counts_by_label": vertex_counts,
            "edge_counts_by_label": edge_counts,
            "vertex_labels": vertex_labels,
            "edge_labels": edge_labels,
        }
