# Codebase Modularization and Enhancement Plan

This document outlines strategies for making the NPS simulation framework more modular and easier to modify. The goal is to improve maintainability, extensibility, and flexibility while preserving the core functionality.

## 1. Configuration Management

### Current State
- Configuration parameters are scattered across multiple files
- Hard-coded values in various modules
- Limited runtime configuration options

### Proposed Changes
1. Create a centralized configuration system:
   ```python
   # config/config.py
   class SimulationConfig:
       def __init__(self):
           self.priority_schemes = ["NPS", "SRTF", "LRTF", "FCFS"]
           self.agent_range = range(3, 10)
           self.default_ceiling_value = 2.5
           self.default_burn_in = 0
           self.default_simulation_days = 365
           # ... other configurable parameters
   ```

2. Implement configuration inheritance:
   ```python
   class CustomConfig(SimulationConfig):
       def __init__(self):
           super().__init__()
           self.priority_schemes = ["NPS", "CUSTOM"]
           self.agent_range = range(5, 15)
   ```

## 2. Dependency Injection

### Current State
- Direct imports and instantiation of dependencies
- Tight coupling between components
- Difficult to swap implementations

### Proposed Changes
1. Create interfaces for core components:
   ```python
   # interfaces/priority_scheme.py
   from abc import ABC, abstractmethod
   
   class PriorityScheme(ABC):
       @abstractmethod
       def calculate_priority(self, case):
           pass
   ```

2. Implement dependency injection:
   ```python
   class Simulation:
       def __init__(self, priority_scheme: PriorityScheme, 
                    case_arrival: CaseArrival,
                    queue_manager: QueueManager):
           self.priority_scheme = priority_scheme
           self.case_arrival = case_arrival
           self.queue_manager = queue_manager
   ```

## 3. Model Abstraction

### Current State
- Direct implementation of mathematical models
- Hard-coded parameters
- Limited flexibility in model selection

### Proposed Changes
1. Create abstract model interfaces:
   ```python
   # models/base.py
   class NPSModel(ABC):
       @abstractmethod
       def predict_nps(self, case_data):
           pass
   
   class ThroughputModel(ABC):
       @abstractmethod
       def predict_throughput(self, case_data):
           pass
   ```

2. Implement model factories:
   ```python
   class ModelFactory:
       @staticmethod
       def create_nps_model(model_type: str) -> NPSModel:
           if model_type == "default":
               return DefaultNPSModel()
           elif model_type == "custom":
               return CustomNPSModel()
   ```

## 4. Event System

### Current State
- Direct method calls between components
- Limited event tracking
- Difficult to add new event types

### Proposed Changes
1. Implement an event bus:
   ```python
   class EventBus:
       def __init__(self):
           self.subscribers = {}
           
       def subscribe(self, event_type, callback):
           if event_type not in self.subscribers:
               self.subscribers[event_type] = []
           self.subscribers[event_type].append(callback)
           
       def publish(self, event_type, data):
           for callback in self.subscribers.get(event_type, []):
               callback(data)
   ```

2. Define event types:
   ```python
   class EventTypes:
       CASE_ARRIVAL = "case_arrival"
       CASE_ASSIGNMENT = "case_assignment"
       CASE_COMPLETION = "case_completion"
       QUEUE_UPDATE = "queue_update"
   ```

## 5. Data Management

### Current State
- Direct file I/O operations
- Scattered data storage logic
- Limited data validation

### Proposed Changes
1. Create data access layer:
   ```python
   class DataAccess:
       def __init__(self, storage_backend):
           self.storage = storage_backend
           
       def save_simulation_result(self, result):
           self.storage.save(result)
           
       def load_simulation_result(self, id):
           return self.storage.load(id)
   ```

2. Implement storage backends:
   ```python
   class StorageBackend(ABC):
       @abstractmethod
       def save(self, data):
           pass
       
       @abstractmethod
       def load(self, id):
           pass
   ```

## 6. Testing Infrastructure

### Current State
- Limited test coverage
- Difficult to test individual components
- No clear testing strategy

### Proposed Changes
1. Implement unit testing framework:
   ```python
   # tests/test_priority_scheme.py
   class TestPriorityScheme(unittest.TestCase):
       def setUp(self):
           self.scheme = NPSPriorityScheme()
           
       def test_priority_calculation(self):
           case = MockCase(nps_score=8)
           priority = self.scheme.calculate_priority(case)
           self.assertEqual(priority, 0.5)
   ```

2. Add integration tests:
   ```python
   # tests/test_simulation.py
   class TestSimulation(unittest.TestCase):
       def test_end_to_end_simulation(self):
           config = TestConfig()
           simulation = Simulation(config)
           results = simulation.run()
           self.assertValidResults(results)
   ```

## Implementation Strategy

1. **Phase 1: Configuration and Interfaces**
   - Implement configuration management
   - Create core interfaces
   - Add dependency injection

2. **Phase 2: Model Abstraction**
   - Refactor existing models
   - Implement model factories
   - Add model validation

3. **Phase 3: Event System**
   - Implement event bus
   - Refactor existing components to use events
   - Add event logging

4. **Phase 4: Data Management**
   - Implement data access layer
   - Add storage backends
   - Implement data validation

5. **Phase 5: Testing**
   - Add unit tests
   - Implement integration tests
   - Add test coverage reporting

## Benefits

1. **Increased Flexibility**
   - Easy to swap implementations
   - Configurable at runtime
   - Extensible architecture

2. **Improved Maintainability**
   - Clear separation of concerns
   - Reduced code duplication
   - Better error handling

3. **Enhanced Testing**
   - Testable components
   - Clear testing strategy
   - Better quality assurance

4. **Better Documentation**
   - Clear interfaces
   - Explicit dependencies
   - Better code organization

## Migration Strategy

1. Create new interfaces and abstract classes
2. Implement new components alongside existing ones
3. Gradually migrate functionality to new components
4. Add tests for new components
5. Deprecate old implementations
6. Remove deprecated code

This plan provides a roadmap for making the codebase more modular and easier to modify while maintaining its current functionality. Each phase can be implemented independently, allowing for gradual improvement of the codebase. 