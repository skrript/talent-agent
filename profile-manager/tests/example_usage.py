"""Example implementation showing how to use the GraphDB integration tests.

This file demonstrates how to create tests for a new GraphDB implementation
by inheriting from the abstract test classes.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

# Add the mcp-server directory to the path
project_root = Path(__file__).parent.parent
mcp_server_path = project_root / "mcp-server"
sys.path.insert(0, str(mcp_server_path))

from graph.base import GraphDB, GremlinQueryInterface, GraphOperationsInterface
from tests.test_interfaces import (
    GraphDBTestBase,
    GremlinQueryInterfaceTestBase, 
    GraphOperationsInterfaceTestBase
)


# Example mock implementation for demonstration
class MockGraphDBImplementation(GraphDB, GremlinQueryInterface, GraphOperationsInterface):
    """Example GraphDB implementation using mocks for demonstration."""
    
    def __init__(self):
        self._connected = False
        self._data = {"vertices": {}, "edges": {}}
        self._next_id = 1
    
    async def _connect(self) -> None:
        """Mock connection."""
        self._connected = True
    
    async def _disconnect(self) -> None:
        """Mock disconnection."""
        self._connected = False
    
    async def __aenter__(self):
        await self._connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._disconnect()
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Mock query execution."""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        # Simple mock implementation
        if "count" in query.lower():
            return [{"count": len(self._data["vertices"])}]
        elif "hasLabel" in query.lower():
            return [{"id": "1", "label": "Person"}]
        elif "invalid" in query.lower():
            raise Exception("Query failed")
        else:
            return []
    
    async def execute_traversal(self, traversal_lambda) -> List[Dict[str, Any]]:
        """Mock traversal execution."""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        # Check if the lambda contains invalidMethod (for testing error handling)
        import inspect
        source = inspect.getsource(traversal_lambda)
        if "invalidMethod" in source:
            raise Exception("Traversal failed")
        
        # Mock traversal result - just return mock data
        return [{"id": "1", "label": "Person"}]
    
    async def add_vertex(self, label: str, properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock vertex addition."""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        vertex_id = f"vertex-{self._next_id}"
        self._next_id += 1
        
        vertex = {
            "id": vertex_id,
            "label": label,
            "properties": properties or {}
        }
        
        self._data["vertices"][vertex_id] = vertex
        return vertex
    
    async def add_edge(self, from_vertex_id: str, to_vertex_id: str, label: str, properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock edge addition."""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        edge_id = f"edge-{self._next_id}"
        self._next_id += 1
        
        edge = {
            "id": edge_id,
            "label": label,
            "from": from_vertex_id,
            "to": to_vertex_id,
            "properties": properties or {}
        }
        
        self._data["edges"][edge_id] = edge
        return edge
    
    async def get_vertex(self, vertex_id: str) -> Dict[str, Any] | None:
        """Mock vertex retrieval."""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        return self._data["vertices"].get(vertex_id)
    
    async def get_schema(self) -> Dict[str, Any]:
        """Mock schema retrieval."""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        vertex_labels = set()
        edge_labels = set()
        
        for vertex in self._data["vertices"].values():
            vertex_labels.add(vertex["label"])
        
        for edge in self._data["edges"].values():
            edge_labels.add(edge["label"])
        
        return {
            "vertices": {
                "labels": list(vertex_labels),
                "properties": {}
            },
            "edges": {
                "labels": list(edge_labels),
                "properties": {},
                "connections": {}
            }
        }
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Mock statistics retrieval."""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        vertex_counts = {}
        edge_counts = {}
        
        for vertex in self._data["vertices"].values():
            label = vertex["label"]
            vertex_counts[label] = vertex_counts.get(label, 0) + 1
        
        for edge in self._data["edges"].values():
            label = edge["label"]
            edge_counts[label] = edge_counts.get(label, 0) + 1
        
        return {
            "total_vertices": len(self._data["vertices"]),
            "total_edges": len(self._data["edges"]),
            "vertex_counts_by_label": vertex_counts,
            "edge_counts_by_label": edge_counts,
            "vertex_labels": list(vertex_counts.keys()),
            "edge_labels": list(edge_counts.keys())
        }


# Example test class that inherits from abstract test classes
class TestMockGraphDBImplementation(GraphDBTestBase, GremlinQueryInterfaceTestBase, GraphOperationsInterfaceTestBase):
    """Test the mock GraphDB implementation using the abstract test classes."""
    
    def get_graphdb_instance(self) -> MockGraphDBImplementation:
        """Return a mock GraphDB instance for testing."""
        return MockGraphDBImplementation()
    
    def get_gremlin_instance(self) -> MockGraphDBImplementation:
        """Return a mock Gremlin interface instance for testing."""
        return MockGraphDBImplementation()
    
    def get_graph_ops_instance(self) -> MockGraphDBImplementation:
        """Return a mock graph operations instance for testing."""
        return MockGraphDBImplementation()
    
    async def setup_test_data(self, instance: MockGraphDBImplementation):
        """Set up test data in the mock implementation."""
        # Add some test vertices
        await instance.add_vertex("Person", {"name": "Alice", "age": 30})
        await instance.add_vertex("Person", {"name": "Bob", "age": 25})
        await instance.add_vertex("Company", {"name": "Acme Corp"})
        
        # Add some test edges
        await instance.add_edge("vertex-1", "vertex-2", "KNOWS", {"since": "2020-01-01"})
    
    async def cleanup_test_data(self, instance: MockGraphDBImplementation):
        """Clean up test data from the mock implementation."""
        # Clear all data
        instance._data = {"vertices": {}, "edges": {}}
        instance._next_id = 1
    
    # Additional implementation-specific tests
    @pytest.mark.asyncio
    async def test_mock_specific_functionality(self):
        """Test functionality specific to the mock implementation."""
        instance = self.get_graphdb_instance()
        
        async with instance:
            # Test that our mock implementation works correctly
            vertex = await instance.add_vertex("TestVertex", {"test": "value"})
            assert vertex["id"] == "vertex-1"
            assert vertex["label"] == "TestVertex"
            assert vertex["properties"]["test"] == "value"
            
            # Test that data persists
            retrieved = await instance.get_vertex("vertex-1")
            assert retrieved == vertex
            
            # Test edge creation
            edge = await instance.add_edge("vertex-1", "vertex-2", "TEST_EDGE")
            assert edge["id"] == "edge-2"
            assert edge["from"] == "vertex-1"
            assert edge["to"] == "vertex-2"
    
    @pytest.mark.asyncio
    async def test_mock_error_handling(self):
        """Test error handling in the mock implementation."""
        instance = self.get_graphdb_instance()
        
        # Test operations without connection
        with pytest.raises(ConnectionError):
            await instance.add_vertex("Person")
        
        # Test with connection
        async with instance:
            # Should work now
            vertex = await instance.add_vertex("Person")
            assert vertex is not None


# Example of how to test a real implementation
class TestRealGraphDBImplementation(GraphDBTestBase):
    """
    Example of how to test a real GraphDB implementation.
    
    This class shows the pattern for testing actual database implementations.
    Uncomment and modify as needed for your specific implementation.
    """
    
    def get_graphdb_instance(self):
        """Return a real GraphDB instance for testing."""
        # Example: return your actual implementation
        # return MyRealGraphDBImplementation(
        #     host="localhost",
        #     port=8182,
        #     database="test"
        # )
        pytest.skip("Real implementation not available")
    
    @pytest.mark.asyncio
    async def test_real_implementation_specific(self):
        """Test functionality specific to the real implementation."""
        # Add tests specific to your real implementation
        pytest.skip("Real implementation not available")


# Example of how to run specific test categories
class TestExampleCategories:
    """Examples of different test categories."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unit_example(self):
        """Example unit test using mocks."""
        mock_client = MockGraphDBImplementation()
        async with mock_client:
            result = await mock_client.add_vertex("Person")
            assert result["label"] == "Person"
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires real database")
    async def test_integration_example(self):
        """Example integration test requiring real database."""
        # This would test against a real database
        pass
    
    @pytest.mark.slow
    async def test_slow_example(self):
        """Example slow test."""
        # This would be a performance test
        pass


if __name__ == "__main__":
    # Example of how to run tests programmatically
    pytest.main([__file__, "-v"])
