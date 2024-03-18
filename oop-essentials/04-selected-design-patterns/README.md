# Useful Design Patterns for ML Projects

## The Strategy Pattern

Algorithms are typically composed of a set of lower-level algorithms (or sub-routines).
There could be more than one way to perform the tasks these lower-level algorithms attend to.
The *strategy pattern* allows the respective lower-level behaviour to be modified by having the high-level algorithm accept an object of a class representing an abstraction of the lower-level task.

Recall the example from the case study: The `ModelEvaluation` could be parametrised with one or more `Metric` instances,
which perform the actual task of computing an evaluation metric:

```python
class ModelEvaluation:
    def __init__(self, X: pd.DataFrame, y: pd.Series,
            metrics: List[Metric],
            test_size: float = 0.3, shuffle: bool = True, random_state: int = 42):
        ...
```

In accordance with the dependency inversion principle, `ModelEvaluation` depends on the abstraction `Metric`,
which encapsulates the logic to compute a specific metric value.

```python
class Metric(ABC):
    @abstractmethod
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        pass
```

It can then be specialised in many ways and subsequently injected into the evaluation object in order to modify its behaviour accordingly.

```python
class MetricMeanAbsError(Metric):
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        return metrics.mean_absolute_error(y_ground_truth, y_predicted)


class MetricR2(Metric):
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        return metrics.r2_score(y_ground_truth, y_predicted)
```

A new metric can be supported by implementing a new implementation of the `Metric` abstraction and without requiring
modifications to the main algorithm implemented in `ModelEvaluation` (open-closed principle).

## The Factory Pattern

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

Apart from the injection of creation mechanisms, factories can be used as configuration objects which can more readily be stored or transferred than the objects themselves.
If, for instance, the actual object would be difficult to persist for technical reasons, has an unstable representation or is excessively large, it can make sense to use a factory instead of the actual object.

There is also an alternative notion of a factory:
We can make use of **functions as factories**, which, in OOP, are typically implemented as static methods or class methods.
In particular, such factories are frequently used to provide alternative construction mechanisms for objects and use the naming scheme `Class.from_something`.
For instance, we may want to be able to load an `RLAgent` from a persisted pickle file and add a class method `from_pickle_file` to obtain an instance of the class:

```python
class RLAgent(ABC):
    @classmethod
    def from_pickle_file(cls, path: str) -> Self:
        pass
```

## The Registry Pattern

A *registry* is a container of objects that allows us to conveniently retrieve the respective objects, e.g. by name.
We can combine this with the factory pattern and register a collection of factories.
If the set of objects is fixed, we can use an enumeration as the registry.


<hr>

[Next: Making Use of IDE Features](../05-ide-features/README.md)
