# Useful Design Patterns for ML Projects

## Strategy Pattern


## Factory

Algorithms frequently need to dynamically create objects, and we often want the way in which the objects are created to be user-definable.
The *factory pattern* addresses this need.

For instance, we might have the requirement to create models within an algorithm which creates separate models for different countries/regions. 
The algorithm internally splits the data and at some point needs to create a new model, which then is trained only on the data of a particular country.
We thus need a way of injecting the model creation mechanism into the algorithm.
This can be formalised by introducing a factory abstraction,

```python
class ModelFactory(ABC):
    def create_model(self) -> Model:
        pass
```

which then can be specialised to handle the concrete models we want to use:

```python
class MyModelFactory(ModelFactory):
    def __init__(self, config: MyModelConfig):
        self._config = config
    
    def create_model(self):
        ...
```

The factory could then be passed to the learning process

```python
class RegionalLearner:
    def __init__(self, database: Database):
        self.database = database
        ...
    
    def train_models(self, model_factory: ModelFactory) -> List[Model]:
        ...
        for regional_data in self.database.split_regions():
            ...
            model = model_factory.create_model()
            model.fit(regional_data)
            ...
```

As in the above example, a factory typically has at least one `create` method that handles the actual object creation.
The configuration which determines how exactly (i.e. with which parameters) each concrete object is to be generated is typically stored in the attributes of the factory (`_config` in our example).

There are also cases where the factory needs to receive essential parameters from the caller, e.g. 
cases where it wouldn't be possible to specify these parameter at the time of construction of the factory, because they do not exist yet and are generated at some point during the execution of the algorithm that receives the factory.

Consider, for example, the use case of training a reinforcement learning agent, whose construction depends on the environment the agent is to operate in.
The environment is generated somewhere with the RL learning process, and given the environment, we then need to generate a specialised agent and train it. 
In such a case, the RL learning process could receive a factory with a signature as follows,

```python
class RLAgentFactory(ABC):
    def create_agent(self, env: Env) -> RLAgent:
        pass
```

which we could then inject into a training mechanism:

```python
class RLProcess:
    def __init__(self, env_factory: EnvFactory, agent_factory: Agent_factory, ...):
        ...

    def run(self) -> TrainingResult:
        ...
```

We can also make use of **functions as factories**, which, in OOP, are typically implemented as static methods or class methods.
In particular, such factories are frequently used to provide alternative construction mechanisms for objects and use the naming scheme `Class.from_something`.
For instance, we may want to be able to load an `RLAgent` from a persisted pickle file and add a class method `from_pickle_file` to obtain an instance of the class:

```python
class RLAgent(ABC):
    @classmethod
    def from_pickle_file(cls, path: str) -> Self:
        pass
```

## Registry