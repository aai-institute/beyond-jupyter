# Needlessly Externalising Configuration

Parts of the machine learning community have an irrational obsession with external configuration, 
and it causes a lot of unnecessary work without providing any benefits. 
The obsession stems, most likely, from the belief that having external configuration is a "best practice" in general.
We claim that, in many cases, configuration can be replaced with high-level code, which is
  * more flexible,
  * more maintainable,
  * and easier to specify in the first place.

## External Configuration Has Its Place

The purpose of configuration is to enable settings to change without requiring the source code to change. 
This is particularly important if code is executed in different environments and the software needs to adapt to these environments (e.g. by using a different local storage folder, using a different database or different access credentials). 
In such cases, it wouldn't be practical to have to create differently parametrised versions of the software; 
it makes much more sense to make the software access exchangeable configuration instead.

Valid use cases for external configuration are thus cases where the configuration changes dynamically, i.e. different configurations are applied in order to adapt, in particular, to external conditions.

## Static External Configuration Is Questionable

By contrast, the anti-pattern we consider is chiefly concerned with the case where the configuration statically configures an experiment or task. 
Specifically, if configuration serves to configure *what* a process is doing rather than to provide context settings, we have to ask ourselves why it needs to be *external* to the code! 
After all, specifying what do is precisely what programming languages are designed for. 
Therefore, in such a case, we could instead have the configuration in code, and the respective code would never need to change as long as the experiment remained the same. 
If we wanted to change the experiment, changing the source file is no different from changing the configuration file.

If we had several such configurations and thus are defining entirely separate experiments, we might as well use different source files rather than different configuration files. 
The key question to ask is: What are we really gaining by using configuration files rather than source files?
In cases where the configuration is static rather than dynamic, do we really need it?
What are we gaining and what are we losing?

## From Configuration to Code

The primary reason why configuration is deemed necessary is that it is viewed as the only way to make configuration explicit.
It is, of course, desirable to make key settings controllable in a single, well-defined place, and a typical misconception is that external configuration is the best or even the only way to achieve this.
If you have followed the rest of our course material, however, it should be clear that it can also be achieved through high-level abstractions which render task specifications entirely declarative.

Imagine a script such as the following, where each function call stands for a larger piece of code which uses the arguments it receives somewhere:

```python
config = read_config("my_config_file")

initialize()

do_something(config.param1, config.param2)

do_something_in_between()

do_something_else(config.param3)

write_results(config.param4)
```

In this case, the configuration serves to define a single place in which all the relevant parameters can be defined.
We could trivially refactor this as follows and define a function which receives only a configuration object, which in this case, could be a data class (such that all attribute types are well-defined): 

```python
@dataclass
class Config:
    param1: str
    param2: float
    param3: int
    param4: Path

    
def run_experiment(config: Config):
    initialize()
    do_something(config.param1, config.param2)
    do_something_in_between()
    do_something_else(config.param3)
    write_results(config.param4)
``` 

In the corresponding main script, which could replace the configuration file, we would then simply have:

```python

run_experiment(Config(param1=..., param2=..., param3=..., param4=...))
```

In an object-oriented solution, we could go even further and establish a high-level abstraction for the experiment, which encapsulates the configuration and the way in which it is run:

```python
Experiment(param1=..., param2=..., param3=..., param4=...).run_main()
``` 

## Reasons to Prefer Code to Configuration

Under the conditions described above, high-level code has the following advantages over external configuration:

* **ease of definition**

  When writing the configuration for an experiment, you get full IDE support (type hints, auto-completions, etc.), making the configuration easier to specify.
  Furthermore, existing code-based configurations are self-documenting (through types and docstrings in the language) and thus easier to understand (as the respective documentation can be directly requested via the IDE).

* **flexibility**

  A specification in Python code can use the full expressiveness of the Python language and apply all of its control flow structures and other language features, e.g. subtype polymorphism.
  The specification can thus take more complex forms (while still maintaining the declarative character of the specification).

* **static type checking**

  Many specification errors are already apparent at the type level, and using Python (with type annotations) to define the configuration has the advantage of types being checkable via static type checkers (e.g. IDEs or tools such as mypy), greatly reducing the potential for specification errors.

* **maintainability**

  A specification in Python code is easier to maintain, because it can be more safely refactored (automatically) or otherwise changed without the danger of inducing a mismatch between configuration and code.


Configuration has none of these advantages, so the key question to ask is: Is the use of configuration necessary to achieve an elegant solution? In other words: Would not using configuration result in suboptimal design?
Because of the downsides of configuration when compared to specifications in the programming language, we should only ever use configuration if the answer to these questions is a very clear "yes".

### Example: Tianshou High-Level Experiment

Consider this example from the Tianshou reinforcement learning library, which heavily makes use of the flexibility of the Python language:
It uses the builder pattern to flexibly configure an experiment; individual arguments can use subtype polymorphism to achieve vastly different behaviour.
The *entire* code snippet is high-level code which defines the configuration of an experiment.
Extracting the constants from it would *not* suffice as configuration, as this would fail to maintain the same level of flexibility.
(Note, however, that even if it was the case that the constant literals sufficed and we could move them to external configuration, we still would not gain anything by doing so!) 

```python
    
experiment = (
        DQNExperimentBuilder(
            EnvFactoryRegistered(task="CartPole-v1", seed=0, venv_type=VectorEnvType.DUMMY),
            ExperimentConfig(
                persistence_enabled=False,
                watch=True,
                watch_render=1 / 35,
                watch_num_episodes=100,
            ),
            SamplingConfig(
                num_epochs=10,
                step_per_epoch=10000,
                batch_size=64,
                num_train_envs=10,
                num_test_envs=100,
                buffer_size=20000,
                step_per_collect=10,
                update_per_step=1 / 10,
            ),
        )
        .with_dqn_params(
            DQNParams(
                lr=1e-3,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320,
            ),
        )
        .with_model_factory_default(hidden_sizes=(64, 64))
        .with_epoch_train_callback(EpochTrainCallbackDQNSetEps(0.3))
        .with_epoch_test_callback(EpochTestCallbackDQNSetEps(0.0))
        .with_epoch_stop_callback(EpochStopCallbackRewardThreshold(195))
        .build()
    )
    experiment.run()
```

As a simple example, consider how we could, using external configuration, handle variations of this experiment where  

 * `with_model_factory_default` shall not be called at all (because we don't want to specify non-default hidden sizes)
 * `with_model_factory(MyModelFactory())` shall be called instead of `with_model_factory_default`.

The solutions to both problems would not be entirely trivial to handle with configuration and would necessitate a differentiation of the various cases in the code that interprets the configuration. 
When using a Python script, however, the changes would be completely straightforward and clean. 

Most importantly, think, once again, about whether there is anything you concretely gained
by using configuration in the first place; and if anything, is it worth sacrificing the advantages we listed above?
