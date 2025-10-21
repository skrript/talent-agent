"""Abstract test classes for GraphDB interfaces.

These tests can be inherited by concrete implementations to ensure
interface compliance and proper behavior.
"""

import pytest
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable
import sys
from pathlib import Path

# Add the mcp-server directory to the path
project_root = Path(__file__).parent.parent
mcp_server_path = project_root / "mcp-server"
sys.path.insert(0, str(mcp_server_path))

from graph.base import GraphDB, GremlinQueryInterface, GraphOperationsInterface


class GraphDBTestBase(ABC):
    """Abstract base class for testing GraphDB implementations."""
    
    @abstractmethod
    def get_graphdb_instance(self) -> GraphDB:
        """Return an instance of the GraphDB implementation to test."""
        pass
    
    @pytest.mark.asyncio
    async def test_context_manager_enter_exit(self):
        """Test that the GraphDB can be used as an async context manager."""
        graphdb = self.get_graphdb_instance()
        
        # Test entering context
        async with graphdb as db:
            assert db is graphdb
            # Should be connected after entering context
            assert hasattr(db, '_connected')
        
        # Should be disconnected after exiting context
        # Note: We can't directly test _connected state as it might be private
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection and disconnection methods."""
        graphdb = self.get_graphdb_instance()
        
        # Test connection
        await graphdb._connect()
        
        # Test disconnection
        await graphdb._disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_connect_calls(self):
        """Test that multiple connect calls don't cause issues."""
        graphdb = self.get_graphdb_instance()
        
        # First connection
        await graphdb._connect()
        
        # Second connection should not raise error
        await graphdb._connect()
        
        # Cleanup
        await graphdb._disconnect()


class GremlinQueryInterfaceTestBase(ABC):
    """Abstract base class for testing GremlinQueryInterface implementations."""
    
    @abstractmethod
    def get_gremlin_instance(self) -> GremlinQueryInterface:
        """Return an instance of the GremlinQueryInterface implementation to test."""
        pass
    
    @abstractmethod
    async def setup_test_data(self, instance: GremlinQueryInterface):
        """Set up test data in the database."""
        pass
    
    @abstractmethod
    async def cleanup_test_data(self, instance: GremlinQueryInterface):
        """Clean up test data from the database."""
        pass
    
    @pytest.mark.asyncio
    async def test_execute_query_basic(self):
        """Test basic query execution."""
        instance = self.get_gremlin_instance()
        
        async with instance:
            await self.setup_test_data(instance)
            
            try:
                # Test simple query
                results = await instance.execute_query("g.V().count()")
                assert isinstance(results, list)
                
                # Test query with parameters
                results = await instance.execute_query(
                    "g.V().hasLabel(label).count()", 
                    {"label": "Person"}
                )
                assert isinstance(results, list)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_execute_query_with_params(self):
        """Test query execution with parameters."""
        instance = self.get_gremlin_instance()
        
        async with instance:
            await self.setup_test_data(instance)
            
            try:
                # Test with various parameter types
                test_params = {
                    "string_param": "test",
                    "int_param": 42,
                    "float_param": 3.14,
                    "bool_param": True,
                    "list_param": [1, 2, 3]
                }
                
                results = await instance.execute_query(
                    "g.V().hasLabel(label).has('name', name).count()",
                    {"label": "Person", "name": "John Doe"}
                )
                assert isinstance(results, list)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_execute_traversal_basic(self):
        """Test basic traversal execution."""
        instance = self.get_gremlin_instance()
        
        async with instance:
            await self.setup_test_data(instance)
            
            try:
                # Test simple traversal
                traversal_lambda = lambda g: g.V().count()
                results = await instance.execute_traversal(traversal_lambda)
                assert isinstance(results, list)
                
                # Test more complex traversal
                traversal_lambda = lambda g: g.V().hasLabel('Person').limit(5)
                results = await instance.execute_traversal(traversal_lambda)
                assert isinstance(results, list)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_execute_traversal_with_filters(self):
        """Test traversal execution with various filters."""
        instance = self.get_gremlin_instance()
        
        async with instance:
            await self.setup_test_data(instance)
            
            try:
                # Test traversal with property filters
                traversal_lambda = lambda g: g.V().hasLabel('Person').has('age', 30)
                results = await instance.execute_traversal(traversal_lambda)
                assert isinstance(results, list)
                
                # Test traversal with range filters
                from gremlin_python.process.graph_traversal import __
                traversal_lambda = lambda g: g.V().hasLabel('Person').has('age', __.between(20, 40))
                results = await instance.execute_traversal(traversal_lambda)
                assert isinstance(results, list)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self):
        """Test error handling for invalid queries."""
        instance = self.get_gremlin_instance()
        
        async with instance:
            # Test invalid query
            with pytest.raises(Exception):
                await instance.execute_query("invalid gremlin query")
            
            # Test invalid traversal
            with pytest.raises(Exception):
                invalid_lambda = lambda g: g.invalidMethod()
                await instance.execute_traversal(invalid_lambda)


class GraphOperationsInterfaceTestBase(ABC):
    """Abstract base class for testing GraphOperationsInterface implementations."""
    
    @abstractmethod
    def get_graph_ops_instance(self) -> GraphOperationsInterface:
        """Return an instance of the GraphOperationsInterface implementation to test."""
        pass
    
    @abstractmethod
    async def setup_test_data(self, instance: GraphOperationsInterface):
        """Set up test data in the database."""
        pass
    
    @abstractmethod
    async def cleanup_test_data(self, instance: GraphOperationsInterface):
        """Clean up test data from the database."""
        pass
    
    @pytest.mark.asyncio
    async def test_add_vertex_basic(self):
        """Test basic vertex addition."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                # Test adding vertex with label only
                result = await instance.add_vertex("Person")
                assert isinstance(result, dict)
                assert "id" in result or "label" in result
                
                # Test adding vertex with properties
                properties = {"name": "John Doe", "age": 30}
                result = await instance.add_vertex("Person", properties)
                assert isinstance(result, dict)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_add_vertex_with_properties(self):
        """Test vertex addition with various property types."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                # Test different property types
                properties = {
                    "string_prop": "test",
                    "int_prop": 42,
                    "float_prop": 3.14,
                    "bool_prop": True,
                    "list_prop": [1, 2, 3],
                    "dict_prop": {"nested": "value"}
                }
                
                result = await instance.add_vertex("TestVertex", properties)
                assert isinstance(result, dict)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_add_edge_basic(self):
        """Test basic edge addition."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                # Test adding edge between vertices
                result = await instance.add_edge(
                    "vertex-1", "vertex-2", "KNOWS"
                )
                assert isinstance(result, dict)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_add_edge_with_properties(self):
        """Test edge addition with properties."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                # Test adding edge with properties
                properties = {"weight": 0.8, "since": "2020-01-01"}
                result = await instance.add_edge(
                    "vertex-1", "vertex-2", "KNOWS", properties
                )
                assert isinstance(result, dict)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_get_vertex(self):
        """Test vertex retrieval."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                # Test getting existing vertex
                result = await instance.get_vertex("vertex-1")
                assert isinstance(result, dict) or result is None
                
                # Test getting non-existent vertex
                result = await instance.get_vertex("non-existent")
                assert result is None
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_get_schema(self):
        """Test schema retrieval."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                schema = await instance.get_schema()
                assert isinstance(schema, dict)
                
                # Check schema structure
                assert "vertices" in schema or "edges" in schema
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_get_graph_statistics(self):
        """Test graph statistics retrieval."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                stats = await instance.get_graph_statistics()
                assert isinstance(stats, dict)
                
                # Check that stats contain expected keys
                expected_keys = ["total_vertices", "total_edges"]
                for key in expected_keys:
                    if key in stats:
                        assert isinstance(stats[key], int)
                        assert stats[key] >= 0
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_operations_without_connection(self):
        """Test that operations raise appropriate errors when not connected."""
        instance = self.get_graph_ops_instance()
        
        # Test operations without connection
        with pytest.raises(Exception):  # Should be ConnectionError or similar
            await instance.add_vertex("Person")
        
        with pytest.raises(Exception):
            await instance.add_edge("v1", "v2", "KNOWS")
        
        with pytest.raises(Exception):
            await instance.get_vertex("v1")
        
        with pytest.raises(Exception):
            await instance.get_schema()
        
        with pytest.raises(Exception):
            await instance.get_graph_statistics()
