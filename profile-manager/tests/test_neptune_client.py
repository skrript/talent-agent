"""Concrete test implementations for NeptuneClient.

These tests inherit from the abstract test classes and provide
concrete implementations for testing the NeptuneClient specifically.
"""

import pytest
import os
from typing import Any, Dict, List
import sys
from pathlib import Path

# Add the mcp-server directory to the path
project_root = Path(__file__).parent.parent
mcp_server_path = project_root / "mcp-server"
sys.path.insert(0, str(mcp_server_path))

from graph.aws_neptune import NeptuneClient
from tests.test_interfaces import (
    GraphDBTestBase, 
    GremlinQueryInterfaceTestBase, 
    GraphOperationsInterfaceTestBase
)


class TestNeptuneClientGraphDB(GraphDBTestBase):
    """Test NeptuneClient GraphDB functionality."""
    
    def get_graphdb_instance(self) -> NeptuneClient:
        """Return a NeptuneClient instance for testing."""
        # Use environment variables or test defaults
        endpoint = os.getenv("NEPTUNE_ENDPOINT", "localhost")
        port = int(os.getenv("NEPTUNE_PORT", "8182"))
        
        return NeptuneClient(
            endpoint=endpoint,
            port=port,
            timeout=10,  # Shorter timeout for tests
            max_workers=2  # Fewer workers for tests
        )
    
    @pytest.mark.asyncio
    async def test_neptune_specific_connection_properties(self):
        """Test Neptune-specific connection properties."""
        client = self.get_graphdb_instance()
        
        # Test connection string generation
        expected_conn_str = f"wss://{client.endpoint}:{client.port}/gremlin"
        assert client.conn_str == expected_conn_str
        
        # Test default values
        assert client.port == 8182
        assert client.traversal_source == "g"
        assert client.timeout == 30
        assert client.max_workers == 4


class TestNeptuneClientGremlinQuery(GremlinQueryInterfaceTestBase):
    """Test NeptuneClient Gremlin query functionality."""
    
    def get_gremlin_instance(self) -> NeptuneClient:
        """Return a NeptuneClient instance for testing."""
        endpoint = os.getenv("NEPTUNE_ENDPOINT", "localhost")
        port = int(os.getenv("NEPTUNE_PORT", "8182"))
        
        return NeptuneClient(
            endpoint=endpoint,
            port=port,
            timeout=10,
            max_workers=2
        )
    
    async def setup_test_data(self, instance: NeptuneClient):
        """Set up test data in Neptune."""
        # Add test vertices
        await instance.execute_query(
            "g.addV('Person').property('name', 'John Doe').property('age', 30).next()"
        )
        await instance.execute_query(
            "g.addV('Person').property('name', 'Jane Smith').property('age', 25).next()"
        )
        await instance.execute_query(
            "g.addV('Company').property('name', 'Acme Corp').property('industry', 'Tech').next()"
        )
        
        # Add test edges
        await instance.execute_query(
            "g.V().has('name', 'John Doe').addE('WORKS_FOR').to(g.V().has('name', 'Acme Corp')).next()"
        )
    
    async def cleanup_test_data(self, instance: NeptuneClient):
        """Clean up test data from Neptune."""
        # Remove test data
        await instance.execute_query("g.V().hasLabel('Person').drop()")
        await instance.execute_query("g.V().hasLabel('Company').drop()")
        await instance.execute_query("g.E().hasLabel('WORKS_FOR').drop()")
    
    @pytest.mark.asyncio
    async def test_neptune_specific_query_features(self):
        """Test Neptune-specific query features."""
        instance = self.get_gremlin_instance()
        
        async with instance:
            await self.setup_test_data(instance)
            
            try:
                # Test Neptune's support for complex traversals
                results = await instance.execute_query(
                    "g.V().hasLabel('Person').out('WORKS_FOR').hasLabel('Company').values('name')"
                )
                assert isinstance(results, list)
                
                # Test Neptune's support for aggregations
                results = await instance.execute_query(
                    "g.V().hasLabel('Person').groupCount().by(label())"
                )
                assert isinstance(results, list)
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_neptune_result_normalization(self):
        """Test Neptune-specific result normalization."""
        instance = self.get_gremlin_instance()
        
        async with instance:
            await self.setup_test_data(instance)
            
            try:
                # Test that results are properly normalized
                results = await instance.execute_query("g.V().hasLabel('Person').elementMap()")
                assert isinstance(results, list)
                
                for result in results:
                    assert isinstance(result, dict)
                
            finally:
                await self.cleanup_test_data(instance)


class TestNeptuneClientGraphOperations(GraphOperationsInterfaceTestBase):
    """Test NeptuneClient graph operations functionality."""
    
    def get_graph_ops_instance(self) -> NeptuneClient:
        """Return a NeptuneClient instance for testing."""
        endpoint = os.getenv("NEPTUNE_ENDPOINT", "localhost")
        port = int(os.getenv("NEPTUNE_PORT", "8182"))
        
        return NeptuneClient(
            endpoint=endpoint,
            port=port,
            timeout=10,
            max_workers=2
        )
    
    async def setup_test_data(self, instance: NeptuneClient):
        """Set up test data in Neptune."""
        # Add test vertices using the interface methods
        vertex1 = await instance.add_vertex("Person", {"name": "John Doe", "age": 30})
        vertex2 = await instance.add_vertex("Person", {"name": "Jane Smith", "age": 25})
        company = await instance.add_vertex("Company", {"name": "Acme Corp", "industry": "Tech"})
        
        # Store vertex IDs for edge creation
        self.vertex1_id = vertex1.get("id") or "vertex-1"
        self.vertex2_id = vertex2.get("id") or "vertex-2"
        self.company_id = company.get("id") or "company-1"
    
    async def cleanup_test_data(self, instance: NeptuneClient):
        """Clean up test data from Neptune."""
        # Remove all test data
        await instance.execute_query("g.V().hasLabel('Person').drop()")
        await instance.execute_query("g.V().hasLabel('Company').drop()")
        await instance.execute_query("g.E().hasLabel('WORKS_FOR').drop()")
    
    @pytest.mark.asyncio
    async def test_neptune_specific_vertex_operations(self):
        """Test Neptune-specific vertex operations."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                # Test Neptune's additional vertex methods
                vertex_labels = await instance.get_vertex_labels()
                assert isinstance(vertex_labels, list)
                assert "Person" in vertex_labels
                assert "Company" in vertex_labels
                
                # Test vertex properties retrieval
                vertex_props = await instance.get_vertex_properties("Person")
                assert isinstance(vertex_props, dict)
                assert "Person" in vertex_props
                
                # Test vertex counting
                person_count = await instance.get_vertex_count("Person")
                assert isinstance(person_count, int)
                assert person_count >= 0
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_neptune_specific_edge_operations(self):
        """Test Neptune-specific edge operations."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                # Add edge using the interface method
                edge = await instance.add_edge(
                    self.vertex1_id, 
                    self.company_id, 
                    "WORKS_FOR",
                    {"since": "2020-01-01", "role": "Engineer"}
                )
                assert isinstance(edge, dict)
                
                # Test Neptune's additional edge methods
                edge_labels = await instance.get_edge_labels()
                assert isinstance(edge_labels, list)
                assert "WORKS_FOR" in edge_labels
                
                # Test edge properties retrieval
                edge_props = await instance.get_edge_properties("WORKS_FOR")
                assert isinstance(edge_props, dict)
                assert "WORKS_FOR" in edge_props
                
                # Test edge counting
                edge_count = await instance.get_edge_count("WORKS_FOR")
                assert isinstance(edge_count, int)
                assert edge_count >= 0
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_neptune_comprehensive_schema(self):
        """Test Neptune's comprehensive schema retrieval."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                # Add edge for schema testing
                await instance.add_edge(
                    self.vertex1_id, 
                    self.company_id, 
                    "WORKS_FOR",
                    {"since": "2020-01-01"}
                )
                
                schema = await instance.get_schema()
                assert isinstance(schema, dict)
                
                # Check schema structure
                assert "vertices" in schema
                assert "edges" in schema
                
                vertices = schema["vertices"]
                assert "labels" in vertices
                assert "properties" in vertices
                
                edges = schema["edges"]
                assert "labels" in edges
                assert "properties" in edges
                assert "connections" in edges
                
                # Verify specific content
                assert "Person" in vertices["labels"]
                assert "Company" in vertices["labels"]
                assert "WORKS_FOR" in edges["labels"]
                
            finally:
                await self.cleanup_test_data(instance)
    
    @pytest.mark.asyncio
    async def test_neptune_comprehensive_statistics(self):
        """Test Neptune's comprehensive statistics."""
        instance = self.get_graph_ops_instance()
        
        async with instance:
            try:
                await self.setup_test_data(instance)
                
                # Add edge for statistics testing
                await instance.add_edge(
                    self.vertex1_id, 
                    self.company_id, 
                    "WORKS_FOR"
                )
                
                stats = await instance.get_graph_statistics()
                assert isinstance(stats, dict)
                
                # Check required statistics
                required_keys = [
                    "total_vertices", "total_edges", 
                    "vertex_counts_by_label", "edge_counts_by_label",
                    "vertex_labels", "edge_labels"
                ]
                
                for key in required_keys:
                    assert key in stats
                
                # Verify counts are reasonable
                assert stats["total_vertices"] >= 0
                assert stats["total_edges"] >= 0
                assert isinstance(stats["vertex_counts_by_label"], dict)
                assert isinstance(stats["edge_counts_by_label"], dict)
                
            finally:
                await self.cleanup_test_data(instance)


# Integration test class that combines all interfaces
class TestNeptuneClientIntegration:
    """Integration tests for NeptuneClient covering all interfaces."""
    
    @pytest.fixture
    def neptune_client(self):
        """Fixture providing a NeptuneClient instance."""
        endpoint = os.getenv("NEPTUNE_ENDPOINT", "localhost")
        port = int(os.getenv("NEPTUNE_PORT", "8182"))
        
        return NeptuneClient(
            endpoint=endpoint,
            port=port,
            timeout=10,
            max_workers=2
        )
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, neptune_client):
        """Test a complete workflow using all interfaces."""
        async with neptune_client:
            try:
                # Setup: Add vertices
                person1 = await neptune_client.add_vertex(
                    "Person", 
                    {"name": "Alice", "age": 28, "department": "Engineering"}
                )
                person2 = await neptune_client.add_vertex(
                    "Person", 
                    {"name": "Bob", "age": 32, "department": "Engineering"}
                )
                project = await neptune_client.add_vertex(
                    "Project", 
                    {"name": "Alpha Project", "status": "active"}
                )
                
                # Add edges
                await neptune_client.add_edge(
                    person1["id"], person2["id"], "COLLABORATES_WITH",
                    {"since": "2023-01-01", "frequency": "daily"}
                )
                await neptune_client.add_edge(
                    person1["id"], project["id"], "WORKS_ON",
                    {"role": "lead", "hours_per_week": 40}
                )
                
                # Query using Gremlin interface
                collaborators = await neptune_client.execute_query(
                    "g.V().has('name', 'Alice').out('COLLABORATES_WITH').values('name')"
                )
                assert isinstance(collaborators, list)
                
                # Query using traversal interface
                projects = await neptune_client.execute_traversal(
                    lambda g: g.V().has('name', 'Alice').out('WORKS_ON').values('name')
                )
                assert isinstance(projects, list)
                
                # Get comprehensive information
                schema = await neptune_client.get_schema()
                stats = await neptune_client.get_graph_statistics()
                
                assert isinstance(schema, dict)
                assert isinstance(stats, dict)
                
                # Verify the data we created
                assert stats["total_vertices"] >= 3
                assert stats["total_edges"] >= 2
                
            finally:
                # Cleanup
                await neptune_client.execute_query("g.V().drop()")
                await neptune_client.execute_query("g.E().drop()")
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, neptune_client):
        """Test error handling across all interfaces."""
        async with neptune_client:
            # Test invalid operations
            with pytest.raises(Exception):
                await neptune_client.execute_query("invalid query")
            
            with pytest.raises(Exception):
                await neptune_client.execute_traversal(lambda g: g.invalidMethod())
            
            # Test operations on non-existent data
            result = await neptune_client.get_vertex("non-existent-id")
            assert result is None
            
            # Test edge creation with non-existent vertices
            with pytest.raises(Exception):
                await neptune_client.add_edge("non-existent-1", "non-existent-2", "TEST")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, neptune_client):
        """Test concurrent operations across interfaces."""
        import asyncio
        
        async with neptune_client:
            try:
                # Create multiple vertices concurrently
                tasks = []
                for i in range(5):
                    task = neptune_client.add_vertex(
                        "TestVertex", 
                        {"id": i, "name": f"Test-{i}"}
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                assert len(results) == 5
                
                # Create multiple edges concurrently
                edge_tasks = []
                for i in range(4):
                    task = neptune_client.add_edge(
                        results[i]["id"], 
                        results[i + 1]["id"], 
                        "CONNECTS_TO"
                    )
                    edge_tasks.append(task)
                
                edge_results = await asyncio.gather(*edge_tasks)
                assert len(edge_results) == 4
                
                # Verify final state
                stats = await neptune_client.get_graph_statistics()
                assert stats["total_vertices"] >= 5
                assert stats["total_edges"] >= 4
                
            finally:
                # Cleanup
                await neptune_client.execute_query("g.V().hasLabel('TestVertex').drop()")
                await neptune_client.execute_query("g.E().hasLabel('CONNECTS_TO').drop()")
