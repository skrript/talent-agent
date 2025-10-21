# GraphDB Integration Tests

This directory contains comprehensive integration tests for the GraphDB interfaces defined in `mcp-server/graph/base.py`. These tests are designed to validate any implementation of the GraphDB interfaces, ensuring they work correctly across different database backends.

## Test Structure

### Core Test Files

- **`test_interfaces.py`** - Abstract test classes that define the contract for each interface
- **`test_neptune_client.py`** - Concrete implementations for testing NeptuneClient
- **`test_mock_integration.py`** - Mock-based tests for interface compliance without database
- **`test_utilities.py`** - Test utilities, helpers, and fixtures
- **`conftest.py`** - Pytest configuration and shared fixtures

### Test Categories

1. **Unit Tests** (`@pytest.mark.unit`) - Use mocks, no database required
2. **Integration Tests** (`@pytest.mark.integration`) - Require actual database connection
3. **Slow Tests** (`@pytest.mark.slow`) - Performance and load tests
4. **Neptune Tests** (`@pytest.mark.neptune`) - Specific to Neptune database

## Running Tests

### Unit Tests Only (Recommended for Development)
```bash
# Run all unit tests (mock-based)
pytest -m unit

# Run specific test file
pytest tests/test_mock_integration.py

# Run with verbose output
pytest -m unit -v
```

### Integration Tests (Requires Database)
```bash
# Set up environment variables
export NEPTUNE_ENDPOINT=your-neptune-endpoint
export NEPTUNE_PORT=8182

# Run integration tests
pytest --run-integration -m integration

# Run Neptune-specific tests
pytest --run-integration -m neptune
```

### All Tests
```bash
# Run everything (unit + integration)
pytest --run-integration --run-slow

# Run with coverage
pytest --run-integration --cov=mcp_server
```

## Test Interface Compliance

### For New GraphDB Implementations

When implementing a new GraphDB backend, follow these steps:

1. **Inherit from Abstract Test Classes**:
   ```python
   from tests.test_interfaces import (
       GraphDBTestBase,
       GremlinQueryInterfaceTestBase, 
       GraphOperationsInterfaceTestBase
   )
   
   class TestMyGraphDB(GraphDBTestBase, GremlinQueryInterfaceTestBase, GraphOperationsInterfaceTestBase):
       def get_graphdb_instance(self):
           return MyGraphDBImplementation()
       
       def get_gremlin_instance(self):
           return MyGraphDBImplementation()
           
       def get_graph_ops_instance(self):
           return MyGraphDBImplementation()
           
       async def setup_test_data(self, instance):
           # Set up test data specific to your implementation
           pass
           
       async def cleanup_test_data(self, instance):
           # Clean up test data
           pass
   ```

2. **Run Interface Compliance Tests**:
   ```bash
   pytest tests/test_interfaces.py::TestMyGraphDB -v
   ```

3. **Add Implementation-Specific Tests**:
   ```python
   class TestMyGraphDBSpecific:
       @pytest.mark.asyncio
       async def test_my_specific_feature(self):
           # Test features unique to your implementation
           pass
   ```

## Test Utilities

### GraphDBTestHelper
Provides utilities for database setup, cleanup, and configuration:
```python
from tests.test_utilities import GraphDBTestHelper

# Create test client
client = GraphDBTestHelper.create_test_neptune_client()

# Ensure clean database
await GraphDBTestHelper.ensure_clean_database(client)

# Create test schema
schema_data = await GraphDBTestHelper.create_test_schema(client)
```

### MockGraphDBFactory
Creates mock implementations for testing interface compliance:
```python
from tests.test_utilities import MockGraphDBFactory

# Create mock implementations
mock_db = MockGraphDBFactory.create_mock_graphdb()
mock_gremlin = MockGraphDBFactory.create_mock_gremlin_interface()
mock_ops = MockGraphDBFactory.create_mock_graph_operations()
```

### TestDataGenerator
Generates test data for various scenarios:
```python
from tests.test_utilities import TestDataGenerator

# Generate test vertices
vertices = TestDataGenerator.generate_vertex_data(count=10)

# Generate test edges
edges = TestDataGenerator.generate_edge_data(vertex_pairs)

# Generate test queries
queries = TestDataGenerator.generate_gremlin_queries()
```

## Environment Configuration

### Required Environment Variables
- `NEPTUNE_ENDPOINT` - Neptune database endpoint
- `NEPTUNE_PORT` - Neptune database port (default: 8182)

### Optional Environment Variables
- `TEST_TIMEOUT` - Test timeout in seconds (default: 30)
- `TEST_MAX_WORKERS` - Maximum workers for tests (default: 2)

## Test Patterns

### 1. Interface Compliance Testing
```python
class TestMyImplementation(GraphDBTestBase):
    def get_graphdb_instance(self):
        return MyImplementation()
    
    # All abstract methods are automatically tested
```

### 2. Mock-Based Testing
```python
@pytest.mark.unit
async def test_with_mocks():
    mock_client = MockGraphDBFactory.create_combined_mock()
    # Test without real database
```

### 3. Integration Testing
```python
@pytest.mark.integration
@pytest.mark.neptune
async def test_with_real_database():
    client = GraphDBTestHelper.create_test_neptune_client()
    async with client:
        # Test with real database
```

### 4. Performance Testing
```python
@pytest.mark.slow
async def test_performance():
    # Load testing, concurrent operations, etc.
```

## Best Practices

1. **Always Clean Up**: Use fixtures or context managers to ensure test data is cleaned up
2. **Use Appropriate Markers**: Mark tests correctly for selective running
3. **Test Error Conditions**: Include tests for error handling and edge cases
4. **Mock External Dependencies**: Use mocks for unit tests to avoid external dependencies
5. **Document Test Purpose**: Each test should have a clear purpose and be well-documented

## Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure Neptune endpoint is accessible and credentials are correct
2. **Timeout Errors**: Increase `TEST_TIMEOUT` environment variable
3. **Cleanup Failures**: Check that test cleanup methods are properly implemented
4. **Mock Issues**: Verify mock configurations match expected interface signatures

### Debug Mode
```bash
# Run with debug output
pytest --run-integration -v -s --tb=long

# Run single test with debug
pytest tests/test_neptune_client.py::TestNeptuneClientIntegration::test_full_workflow -v -s
```

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate pytest markers
3. Include both positive and negative test cases
4. Add docstrings explaining test purpose
5. Update this README if adding new test utilities or patterns
