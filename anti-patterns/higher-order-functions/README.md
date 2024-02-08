# Using Higher-Order Functions Instead of the Strategy Pattern

The use of higher-order functions, i.e. functions that take other functions as parameters, can replace the strategy pattern, where an object supporting the respective function is passed as an argument.

Consider the following alternative implementations of a regression model evaluation use case, where the metric to use shall be parametrisable:
  * [functional implementation](regressor_evaluation_functional.py)

    Because we require additional information on the metric in order to apply it correctly, specifically

      * a string indicating its name for reporting and
      * a flag indicating whether higher values are better,

    the function itself is not sufficient and we thus add three parameters to the evaluation function's interface:

    ```python
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str,
    higher_is_better: bool,
    ```
    
  * [object-oriented implementation](regressor_evaluation_oop.py)

    We define an abstract class `Metric` which handles all the required aspects.

    ```python
    class Metric(ABC):
      @abstractmethod
      def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
          pass

      @abstractmethod
      def get_name(self) -> str:
          pass

      @abstractmethod
      def is_larger_better(self) -> bool:
          pass
    ```

    The evaluation function takes a single argument:

    ```python
    metric: Metric
    ```

Please take a close look at both Python files and think about the implications.

## The Key Issues

The OOP solution has significant advantages over the functional solution: 

- **Encapsulation: The class-based approach allows the algorithm itself and additional information/operations to be appropriately grouped.**

  Grouping the actual computation (method `compute_value`) and the meta-data (name and "larger is better" flag) in a representational unit (abstraction) leads to less cumbersome usage patterns (1 vs. 3 arguments) and furthermore helps to avoid errors:
  Users are not required to think about which combination of parameters is appropriate, as the intended combinations are pre-configured in classes.

  A class is simply a more general concept than a function and therefore is more amenable to complex use cases; classes soundly generalise to cases where an algorithmic method cannot reasonably be reduced to a single function.

- **Discoverability: The abstract base class provides a (searchable) type bound**, which straightforwardly enables discovery of potential implementations.
  We need only to use our IDE's hierarchy view in order to discover all existing implementations of `Metric` that fit the class-based interface:

  ![Hierarchy View](../../oop-essentials/ide-features/res/hierarchy_intellij.png)
 
  By contrast, the functional interface, which specifies `Callable[[np.ndarray, np.ndarray], float]`, supports no such search.
  This can be a significant problem in cases where the functions to apply are not co-located with the function applying them; they could be anywhere within a very large codebase encompassing dozens of modules and thousands of lines of code. Only extensive documentation could avoid severe usability limitations. 
   The OOP solution can do without; the type information is sufficient to discover all existing implementations.

- **Parametrisability: Objects can straightforwardly parametrise their behaviour through attributes**,   
  which alleviates the need for cumbersome *currying*, e.g. through a `lambda` function, a local function (closure) or `functools.partial`.
  
  In our example, we used a lambda function to specify the metric's threshold parameter:
  ```python
  lambda t, u: compute_metric_rel_freq_error_within(t, u, max_error)
  ```
  
  In the object-oriented case, we simply parametrise the object:
  ```python
  MetricRelFreqErrorWithin(max_error)
  ```
  
- **Logging and persistence: Objects have representations which can more readily be logged and stored.** 

  A `lambda` function or anonymous function lacks both, a representation amenable to logging and 
  the possibility of serialisation. 
  While the use of `functools.partial` allows for serialisation (by converting the curried function to an object), it does not have a customizable representation for logging.
  The class, by contrast, gives us full control; we could implement `__str__` or `__repr__` in any way we please - and we optionally have fine-grained control over persistence by implementing `__getstate__` and `__setstate__`.

- **Explicit type relationships**.

  In functional interface specifications, we use `Callable` to declare the type of the function - or a previously defined `Protocol` in the case of a complex function signature involving keyword arguments.
  In either case, the relationship between the declared type and its implementors is, in typical *duck typing* fashion, normally an implicit one.
  By contrast, the object-oriented solution establishes an explicit type relationship.
  An explicit type relationship has the advantage of being checkable, i.e. a static type checker can test whether an interface is indeed implemented correctly (as far as types are concerned), whereas a divergence in duck-typed implementations will remain unnoticed: If a function signature changes (for whatever reason), static type checking will not reveal that the function no longer aligns with the type annotation of the higher-order function it was intended to be used in conjunction with.

## Doubling Down on Encapsulation

These are already important reasons to prefer the object-oriented solution.
But now consider the case where we want to support multiple metrics, as in our [case study in the OOP essentials module](../../oop-essentials/02d-case-study-3-metric-abstraction/README.md).
Since the function alone is not sufficient in our case - we also need the meta-data (name, higher is better) - 
we would have the following options to provide multiple metrics in the functional case:

 * a list of tuples bundling the function with the required metadata: the elements of the tuple can only be accessed by the corresponding index and the docstring would need to explain this in detail.

   ```python
   metric_tuples: List[Tuple[Callable[[np.ndarray, np.ndarray], float], str, bool]],
   ```

 * a list of dictionaries with keys such as "metric_fn", "metric_name", "higher_is_better", i.e 

   ```python
   metrics: List[Dict[str, Any]],
   ```
   
   where we completely lose type information for the values; static type checking cannot take place.
   Again, we would need to explain in detail how to construct the dictionary
   and would need checks to ensure that the format is adhered to.
   Since a dictionary is basically a primitive object with the aforementioned downsides, this would be a most questionable choice.

 * three separate lists
 
   ```python
   metric_fns: List[Callable[[np.ndarray, np.ndarray], float],
   metric_names: List[str],
   higher_is_better_flags: List[bool],
   ```

   which we could iterate over using `zip`. However, we then would need to check for consistency (equal lengths).

Compare these options to the straightforward object-oriented solution, where we simply specify

   ```python
   metrics: List[Metric],
   ```

which gives us full static type checking support and is entirely self-documenting.

## Conclusion

All things considered, we cannot think of many reasons to prefer the functional alternative.
Of course, there are cases where the function is simple, does not need to be parametrised and does not need to be meaningfully logged, etc. - and in such cases, we could think about using a functional interface, as it could be the most concise solution.
As a general rule, however, the object-oriented strategy pattern is the most elegant solution for injecting algorithms.