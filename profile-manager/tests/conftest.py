"""Test configuration and fixtures for GraphDB integration tests."""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the mcp-server directory to the path (it has a hyphen so we need to handle it specially)
mcp_server_path = project_root / "mcp-server"
sys.path.insert(0, str(mcp_server_path))

# Import the graph modules directly
from graph.base import GraphDB, GremlinQueryInterface, GraphOperationsInterface


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring database"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test using mocks"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "neptune: mark test as requiring Neptune database"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "mock" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "neptune" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.neptune)
        elif "interface" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker for tests that might be slow
        if any(keyword in item.nodeid for keyword in ["concurrent", "load", "performance"]):
            item.add_marker(pytest.mark.slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require database connection"
    )
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--neptune-endpoint",
        action="store",
        default="localhost",
        help="Neptune endpoint for integration tests"
    )
    parser.addoption(
        "--neptune-port",
        action="store",
        default="8182",
        help="Neptune port for integration tests"
    )


def pytest_runtest_setup(item):
    """Skip tests based on command line options."""
    if "integration" in item.keywords and not item.config.getoption("--run-integration"):
        pytest.skip("Integration tests skipped. Use --run-integration to run them.")
    
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("Slow tests skipped. Use --run-slow to run them.")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_vertex_data():
    """Sample vertex data for testing."""
    return {
        "label": "Person",
        "properties": {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
    }


@pytest.fixture
def sample_edge_data():
    """Sample edge data for testing."""
    return {
        "from_vertex_id": "person-1",
        "to_vertex_id": "person-2", 
        "label": "KNOWS",
        "properties": {
            "since": "2020-01-01",
            "strength": 0.8
        }
    }


@pytest.fixture
def sample_gremlin_query():
    """Sample Gremlin query for testing."""
    return "g.V().hasLabel('Person').limit(10)"


@pytest.fixture
def sample_traversal_lambda():
    """Sample traversal lambda for testing."""
    return lambda g: g.V().hasLabel('Person').limit(10)


@pytest.fixture
def mock_graphdb():
    """Mock GraphDB implementation for testing interface compliance."""
    mock = MagicMock(spec=GraphDB)
    mock._connect = AsyncMock()
    mock._disconnect = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock()
    return mock


@pytest.fixture
def mock_gremlin_interface():
    """Mock GremlinQueryInterface implementation for testing."""
    mock = MagicMock(spec=GremlinQueryInterface)
    mock.execute_query = AsyncMock(return_value=[{"id": "1", "label": "Person"}])
    mock.execute_traversal = AsyncMock(return_value=[{"id": "1", "label": "Person"}])
    return mock


@pytest.fixture
def mock_graph_operations():
    """Mock GraphOperationsInterface implementation for testing."""
    mock = MagicMock(spec=GraphOperationsInterface)
    mock.add_vertex = AsyncMock(return_value={"id": "vertex-1", "label": "Person"})
    mock.add_edge = AsyncMock(return_value={"id": "edge-1", "label": "KNOWS"})
    mock.get_vertex = AsyncMock(return_value={"id": "vertex-1", "label": "Person"})
    mock.get_schema = AsyncMock(return_value={"vertices": {"labels": ["Person"]}, "edges": {"labels": ["KNOWS"]}})
    mock.get_graph_statistics = AsyncMock(return_value={"total_vertices": 1, "total_edges": 0})
    return mock
