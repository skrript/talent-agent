"""Test utilities and helpers for GraphDB integration tests."""

import asyncio
import os
import pytest
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

# Add the mcp-server directory to the path
project_root = Path(__file__).parent.parent
mcp_server_path = project_root / "mcp-server"
sys.path.insert(0, str(mcp_server_path))

from graph.base import GraphDB, GremlinQueryInterface, GraphOperationsInterface
from graph.aws_neptune import NeptuneClient


class GraphDBTestHelper:
    """Helper class for GraphDB testing utilities."""
    
    @staticmethod
    def get_test_config() -> Dict[str, Any]:
        """Get test configuration from environment variables."""
        return {
            "neptune_endpoint": os.getenv("NEPTUNE_ENDPOINT", "localhost"),
            "neptune_port": int(os.getenv("NEPTUNE_PORT", "8182")),
            "test_timeout": int(os.getenv("TEST_TIMEOUT", "30")),
            "test_max_workers": int(os.getenv("TEST_MAX_WORKERS", "2")),
        }
    
    @staticmethod
    def create_test_neptune_client(**overrides) -> NeptuneClient:
        """Create a NeptuneClient instance for testing with optional overrides."""
        config = GraphDBTestHelper.get_test_config()
        config.update(overrides)
        
        return NeptuneClient(
            endpoint=config["neptune_endpoint"],
            port=config["neptune_port"],
            timeout=config["test_timeout"],
            max_workers=config["test_max_workers"]
        )
    
    @staticmethod
    async def ensure_clean_database(client: Union[NeptuneClient, Any]) -> None:
        """Ensure the database is clean before tests."""
        if hasattr(client, 'execute_query'):
            try:
                # Drop all vertices and edges
                await client.execute_query("g.V().drop()")
                await client.execute_query("g.E().drop()")
            except Exception:
                # Ignore errors if database is already clean
                pass
    
    @staticmethod
    async def create_test_schema(client: Union[NeptuneClient, Any]) -> Dict[str, Any]:
        """Create a standard test schema in the database."""
        schema_data = {
            "vertices": [],
            "edges": []
        }
        
        # Create test vertices
        person1 = await client.add_vertex("Person", {
            "name": "Alice Johnson",
            "age": 30,
            "email": "alice@example.com",
            "department": "Engineering"
        })
        schema_data["vertices"].append(person1)
        
        person2 = await client.add_vertex("Person", {
            "name": "Bob Smith", 
            "age": 28,
            "email": "bob@example.com",
            "department": "Engineering"
        })
        schema_data["vertices"].append(person2)
        
        company = await client.add_vertex("Company", {
            "name": "TechCorp",
            "industry": "Technology",
            "founded": 2010,
            "employees": 500
        })
        schema_data["vertices"].append(company)
        
        project = await client.add_vertex("Project", {
            "name": "Alpha Project",
            "status": "active",
            "budget": 100000,
            "start_date": "2023-01-01"
        })
        schema_data["vertices"].append(project)
        
        # Create test edges
        edge1 = await client.add_edge(
            person1["id"], person2["id"], "COLLABORATES_WITH",
            {"since": "2023-01-01", "frequency": "daily", "strength": 0.8}
        )
        schema_data["edges"].append(edge1)
        
        edge2 = await client.add_edge(
            person1["id"], company["id"], "WORKS_FOR",
            {"since": "2022-06-01", "position": "Senior Engineer", "salary": 120000}
        )
        schema_data["edges"].append(edge2)
        
        edge3 = await client.add_edge(
            person1["id"], project["id"], "LEADS",
            {"since": "2023-01-01", "responsibility": "Technical Lead"}
        )
        schema_data["edges"].append(edge3)
        
        edge4 = await client.add_edge(
            person2["id"], project["id"], "CONTRIBUTES_TO",
            {"since": "2023-02-01", "role": "Developer", "hours_per_week": 40}
        )
        schema_data["edges"].append(edge4)
        
        return schema_data
    
    @staticmethod
    async def cleanup_test_schema(client: Union[NeptuneClient, Any]) -> None:
        """Clean up the test schema from the database."""
        if hasattr(client, 'execute_query'):
            try:
                await client.execute_query("g.V().hasLabel('Person').drop()")
                await client.execute_query("g.V().hasLabel('Company').drop()")
                await client.execute_query("g.V().hasLabel('Project').drop()")
                await client.execute_query("g.E().hasLabel('COLLABORATES_WITH').drop()")
                await client.execute_query("g.E().hasLabel('WORKS_FOR').drop()")
                await client.execute_query("g.E().hasLabel('LEADS').drop()")
                await client.execute_query("g.E().hasLabel('CONTRIBUTES_TO').drop()")
            except Exception:
                # Ignore cleanup errors
                pass


class MockGraphDBFactory:
    """Factory for creating mock GraphDB implementations for testing."""
    
    @staticmethod
    def create_mock_graphdb() -> MagicMock:
        """Create a mock GraphDB implementation."""
        mock = MagicMock(spec=GraphDB)
        mock._connect = AsyncMock()
        mock._disconnect = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock()
        return mock
    
    @staticmethod
    def create_mock_gremlin_interface() -> MagicMock:
        """Create a mock GremlinQueryInterface implementation."""
        mock = MagicMock(spec=GremlinQueryInterface)
        mock.execute_query = AsyncMock(return_value=[
            {"id": "1", "label": "Person", "name": "Alice"},
            {"id": "2", "label": "Person", "name": "Bob"}
        ])
        mock.execute_traversal = AsyncMock(return_value=[
            {"id": "1", "label": "Person", "name": "Alice"}
        ])
        return mock
    
    @staticmethod
    def create_mock_graph_operations() -> MagicMock:
        """Create a mock GraphOperationsInterface implementation."""
        mock = MagicMock(spec=GraphOperationsInterface)
        mock.add_vertex = AsyncMock(return_value={"id": "vertex-1", "label": "Person"})
        mock.add_edge = AsyncMock(return_value={"id": "edge-1", "label": "KNOWS"})
        mock.get_vertex = AsyncMock(return_value={"id": "vertex-1", "label": "Person"})
        mock.get_schema = AsyncMock(return_value={
            "vertices": {"labels": ["Person", "Company"], "properties": {}},
            "edges": {"labels": ["KNOWS", "WORKS_FOR"], "properties": {}}
        })
        mock.get_graph_statistics = AsyncMock(return_value={
            "total_vertices": 2,
            "total_edges": 1,
            "vertex_counts_by_label": {"Person": 1, "Company": 1},
            "edge_counts_by_label": {"KNOWS": 1},
            "vertex_labels": ["Person", "Company"],
            "edge_labels": ["KNOWS"]
        })
        return mock
    
    @staticmethod
    def create_combined_mock() -> MagicMock:
        """Create a mock that implements all interfaces."""
        mock = MagicMock()
        
        # Add GraphDB methods
        mock._connect = AsyncMock()
        mock._disconnect = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock()
        
        # Add GremlinQueryInterface methods
        mock.execute_query = AsyncMock(return_value=[{"id": "1", "label": "Person"}])
        mock.execute_traversal = AsyncMock(return_value=[{"id": "1", "label": "Person"}])
        
        # Add GraphOperationsInterface methods
        mock.add_vertex = AsyncMock(return_value={"id": "vertex-1", "label": "Person"})
        mock.add_edge = AsyncMock(return_value={"id": "edge-1", "label": "KNOWS"})
        mock.get_vertex = AsyncMock(return_value={"id": "vertex-1", "label": "Person"})
        mock.get_schema = AsyncMock(return_value={"vertices": {}, "edges": {}})
        mock.get_graph_statistics = AsyncMock(return_value={"total_vertices": 1, "total_edges": 0})
        
        return mock


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_vertex_data(count: int = 5, label: str = "TestVertex") -> List[Dict[str, Any]]:
        """Generate test vertex data."""
        vertices = []
        for i in range(count):
            vertex = {
                "label": label,
                "properties": {
                    "id": f"test-{i}",
                    "name": f"Test Vertex {i}",
                    "value": i * 10,
                    "active": i % 2 == 0,
                    "tags": [f"tag{j}" for j in range(i % 3 + 1)]
                }
            }
            vertices.append(vertex)
        return vertices
    
    @staticmethod
    def generate_edge_data(
        vertex_pairs: List[tuple], 
        label: str = "CONNECTS_TO"
    ) -> List[Dict[str, Any]]:
        """Generate test edge data."""
        edges = []
        for i, (from_id, to_id) in enumerate(vertex_pairs):
            edge = {
                "from_vertex_id": from_id,
                "to_vertex_id": to_id,
                "label": label,
                "properties": {
                    "weight": 0.5 + (i * 0.1),
                    "created": f"2023-{i+1:02d}-01",
                    "active": True
                }
            }
            edges.append(edge)
        return edges
    
    @staticmethod
    def generate_gremlin_queries() -> List[Dict[str, Any]]:
        """Generate test Gremlin queries."""
        return [
            {
                "query": "g.V().count()",
                "params": None,
                "description": "Count all vertices"
            },
            {
                "query": "g.V().hasLabel(label).count()",
                "params": {"label": "Person"},
                "description": "Count vertices by label"
            },
            {
                "query": "g.V().has('name', name).values('age')",
                "params": {"name": "Alice"},
                "description": "Get age by name"
            },
            {
                "query": "g.V().hasLabel('Person').out('KNOWS').hasLabel('Person').values('name')",
                "params": None,
                "description": "Get friends of people"
            }
        ]
    
    @staticmethod
    def generate_traversal_lambdas():
        """Generate test traversal lambdas."""
        return [
            {
                "lambda": lambda g: g.V().count(),
                "description": "Count all vertices"
            },
            {
                "lambda": lambda g: g.V().hasLabel('Person').limit(10),
                "description": "Get first 10 people"
            },
            {
                "lambda": lambda g: g.V().has('age', __.between(20, 40)),
                "description": "Get people aged 20-40"
            },
            {
                "lambda": lambda g: g.V().hasLabel('Person').out('KNOWS').dedup(),
                "description": "Get unique friends"
            }
        ]


class PerformanceTestHelper:
    """Helper for performance testing utilities."""
    
    @staticmethod
    async def measure_execution_time(coro) -> tuple[Any, float]:
        """Measure execution time of an async operation."""
        import time
        start_time = time.time()
        result = await coro
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    async def run_concurrent_operations(operations: List, max_concurrent: int = 10) -> List[Any]:
        """Run multiple operations concurrently with a limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation
        
        tasks = [run_with_semaphore(op) for op in operations]
        return await asyncio.gather(*tasks)
    
    @staticmethod
    def create_load_test_data(vertex_count: int = 100, edge_count: int = 200):
        """Create data for load testing."""
        vertices = TestDataGenerator.generate_vertex_data(vertex_count)
        
        # Create edges between random vertices
        import random
        edges = []
        for i in range(edge_count):
            from_id = f"test-{random.randint(0, vertex_count-1)}"
            to_id = f"test-{random.randint(0, vertex_count-1)}"
            if from_id != to_id:  # Avoid self-loops
                edges.append({
                    "from_vertex_id": from_id,
                    "to_vertex_id": to_id,
                    "label": "CONNECTS_TO",
                    "properties": {"weight": random.random()}
                })
        
        return vertices, edges


# Pytest fixtures for common test scenarios
@pytest.fixture
def test_helper():
    """Fixture providing GraphDBTestHelper instance."""
    return GraphDBTestHelper()


@pytest.fixture
def mock_factory():
    """Fixture providing MockGraphDBFactory instance."""
    return MockGraphDBFactory()


@pytest.fixture
def data_generator():
    """Fixture providing TestDataGenerator instance."""
    return TestDataGenerator()


@pytest.fixture
def performance_helper():
    """Fixture providing PerformanceTestHelper instance."""
    return PerformanceTestHelper()


@pytest.fixture
async def clean_neptune_client():
    """Fixture providing a clean NeptuneClient for testing."""
    client = GraphDBTestHelper.create_test_neptune_client()
    await GraphDBTestHelper.ensure_clean_database(client)
    
    yield client
    
    # Cleanup after test
    await GraphDBTestHelper.ensure_clean_database(client)
    await client._disconnect()


@pytest.fixture
async def neptune_client_with_schema():
    """Fixture providing a NeptuneClient with test schema."""
    client = GraphDBTestHelper.create_test_neptune_client()
    await GraphDBTestHelper.ensure_clean_database(client)
    
    async with client:
        schema_data = await GraphDBTestHelper.create_test_schema(client)
        yield client, schema_data
        
        # Cleanup after test
        await GraphDBTestHelper.cleanup_test_schema(client)
