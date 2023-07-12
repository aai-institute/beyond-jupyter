from enum import Enum
import functools
from typing import Optional, Callable, Union

from torch.nn import functional as F


class ActivationFunction(Enum):
    NONE = "none"
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LOG_SOFTMAX = "log_softmax"
    SOFTMAX = "softmax"

    @classmethod
    def fromName(cls, name) -> "ActivationFunction":
        for item in cls:
            if item.value.name == name:
                return item
        raise ValueError(f"No function found for name '{name}'")

    def getTorchFunction(self) -> Optional[Callable]:
        return {
                ActivationFunction.NONE: None,
                ActivationFunction.SIGMOID: F.sigmoid,
                ActivationFunction.RELU: F.relu,
                ActivationFunction.TANH: F.tanh,
                ActivationFunction.LOG_SOFTMAX: functools.partial(F.log_softmax, dim=1),
                ActivationFunction.SOFTMAX: functools.partial(F.softmax, dim=1)
            }[self]

    def getName(self) -> str:
        return self.value

    @classmethod
    def torchFunctionFromAny(cls, f: Union[str, "ActivationFunction", Callable, None]) -> Optional[Callable]:
        """
        Gets the torch activation for the given argument

        :param f: either an instance of ActivationFunction, the name of a function from torch.nn.functional or an actual function
        :return: a function that can be applied to tensors (or None)
        """
        if f is None:
            return None
        elif isinstance(f, str):
            try:
                return cls.fromName(f).getTorchFunction()
            except ValueError:
                return getattr(F, f)
        elif isinstance(f, ActivationFunction):
            return f.getTorchFunction()
        elif callable(f):
            return f
        else:
            raise ValueError(f"Could not determine torch function from {f} of type {type(f)}")


class ClassificationOutputMode(Enum):
    PROBABILITIES = "probabilities"
    LOG_PROBABILITIES = "log_probabilities"
    UNNORMALISED_LOG_PROBABILITIES = "unnormalised_log_probabilities"

    @classmethod
    def forActivationFn(cls, fn: Optional[Union[Callable, ActivationFunction]]):
        if isinstance(fn, ActivationFunction):
            fn = fn.getTorchFunction()
        if fn is None:
            return cls.UNNORMALISED_LOG_PROBABILITIES
        if not callable(fn):
            raise ValueError(fn)
        if isinstance(fn, functools.partial):
            fn = fn.func
        name = fn.__name__
        if name in ("sigmoid", "relu", "tanh"):
            raise ValueError(f"The activation function {fn} is not suitable as an output activation function for classification")
        elif name in ("log_softmax",):
            return cls.LOG_PROBABILITIES
        elif name in ("softmax",):
            return cls.PROBABILITIES
        else:
            raise ValueError(f"Unhandled function {fn}")
