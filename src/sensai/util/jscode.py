"""
Utility classes and functions for JavaScript code generation
"""
import json
from abc import abstractmethod, ABC
from typing import Union, Any

import numpy as np

from .string import listString


PythonType = Union[str, int, bool, float]


class JsCode(ABC):
    def __str__(self):
        return self.getJsCode()

    @abstractmethod
    def getJsCode(self):
        pass


class JsCodeLiteral(JsCode):
    def __init__(self, jsCode: str):
        self.jsCode = jsCode

    def getJsCode(self):
        return self.jsCode


class JsValue(JsCode):
    @classmethod
    def fromPython(cls, value: PythonType):
        t = type(value)
        if t == str:
            return cls.stringValue(value)
        elif t == int:
            return cls.intValue(value)
        elif value is None:
            return cls.undefined()
        elif t == bool:
            return cls.boolValue(value)
        elif t in (float, np.float64, np.float):
            return cls.floatValue(value)
        else:
            raise ValueError(f"Unsupported value of type {type(value)}: {value}")

    @classmethod
    def fromValue(cls, value: Union["JsValue", PythonType]):
        if isinstance(value, JsValue):
            return value
        else:
            return cls.fromPython(value)

    def isUndefined(self):
        return self.getJsCode() == "undefined"

    @staticmethod
    def stringValue(s: str):
        s = s.replace('"', r'\"')
        return JsValueLiteral(f'"{s}"')

    @staticmethod
    def intValue(value: int):
        return JsValueLiteral(str(int(value)))

    @staticmethod
    def floatValue(value: Union[float, int]):
        return JsValueLiteral(str(float(value)))

    @staticmethod
    def boolValue(value: bool):
        b = bool(value)
        return JsValueLiteral("true" if b else "false")

    @staticmethod
    def undefined():
        return JsValueLiteral("undefined")

    @staticmethod
    def null(self):
        return JsValueLiteral("null")


class JsValueLiteral(JsValue):
    def __init__(self, jsCode: str):
        self.jsCode = jsCode

    def getJsCode(self):
        return self.jsCode


def jsValue(value: Union[JsValue, PythonType]) -> JsValue:
    return JsValue.fromValue(value)


def jsArgList(*args: Union[JsValue, PythonType], dropTrailingUndefined=True) -> JsCode:
    """
    :param args: arguments that are either JsValue instances or (supported) Python values
    :param dropTrailingUndefined: whether to drop trailing arguments that are undefined/None
    :return: the JsCode
    """
    args = [jsValue(a) for a in args]
    lastIndexToInclude = len(args) - 1
    if dropTrailingUndefined:
        while lastIndexToInclude >= 0 and args[lastIndexToInclude].isUndefined():
            lastIndexToInclude -= 1
    args = args[:lastIndexToInclude+1]
    return JsCodeLiteral(", ".join(map(str, args)))


class JsObject(JsValue):
    def __init__(self):
        self.data = {}

    def add(self, key: str, value: Union[JsValue, PythonType]):
        self.data[key] = jsValue(value)

    def addString(self, key: str, value: str):
        self.data[key] = JsValue.stringValue(value)

    def addCodeLiteral(self, key: str, value: str):
        self.data[key] = value

    def addFloat(self, key: str, value: Union[float, int]):
        self.data[key] = JsValue.floatValue(value)

    def addJson(self, key: str, value: Any):
        """
        :param key: key within the object
        :param value: any Python object which can be converted to JSON
        """
        self.addCodeLiteral(key, json.dumps(value))

    def getJsCode(self):
        return "{" + ", ".join(f'"{k}": {v}' for k, v in self.data.items()) + "}"

    def __len__(self):
        return len(self.data)


class JsClassInstance(JsValueLiteral):
    def __init__(self, className, *args: Union[JsValue, PythonType]):
        argList = jsArgList(*args, dropTrailingUndefined=False)
        super().__init__(f"new {className}({argList})")


class JsList(JsValueLiteral):
    def __init__(self, *values: Union[JsValue, PythonType]):
        super().__init__(listString([jsValue(x) for x in values]))

