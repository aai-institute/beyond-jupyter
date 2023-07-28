"""
Utility classes and functions for JavaScript code generation
"""
import json
from abc import abstractmethod, ABC
from typing import Union, Any

import numpy as np

from .string import list_string


PythonType = Union[str, int, bool, float]


class JsCode(ABC):
    def __str__(self):
        return self.get_js_code()

    @abstractmethod
    def get_js_code(self):
        pass


class JsCodeLiteral(JsCode):
    def __init__(self, js_code: str):
        self.js_code = js_code

    def get_js_code(self):
        return self.js_code


class JsValue(JsCode, ABC):
    @classmethod
    def from_python(cls, value: PythonType):
        t = type(value)
        if t == str:
            return cls.string_value(value)
        elif t == int:
            return cls.int_value(value)
        elif value is None:
            return cls.undefined()
        elif t == bool:
            return cls.bool_value(value)
        elif t in (float, np.float64, np.float):
            return cls.float_value(value)
        else:
            raise ValueError(f"Unsupported value of type {type(value)}: {value}")

    @classmethod
    def from_value(cls, value: Union["JsValue", PythonType]):
        if isinstance(value, JsValue):
            return value
        else:
            return cls.from_python(value)

    def is_undefined(self):
        return self.get_js_code() == "undefined"

    @staticmethod
    def string_value(s: str):
        s = s.replace('"', r'\"')
        return JsValueLiteral(f'"{s}"')

    @staticmethod
    def int_value(value: int):
        return JsValueLiteral(str(int(value)))

    @staticmethod
    def float_value(value: Union[float, int]):
        return JsValueLiteral(str(float(value)))

    @staticmethod
    def bool_value(value: bool):
        b = bool(value)
        return JsValueLiteral("true" if b else "false")

    @staticmethod
    def undefined():
        return JsValueLiteral("undefined")

    @staticmethod
    def null():
        return JsValueLiteral("null")


class JsValueLiteral(JsValue):
    def __init__(self, js_code: str):
        self.js_code = js_code

    def get_js_code(self):
        return self.js_code


def js_value(value: Union[JsValue, PythonType]) -> JsValue:
    return JsValue.from_value(value)


def js_arg_list(*args: Union[JsValue, PythonType], drop_trailing_undefined=True) -> JsCode:
    """
    :param args: arguments that are either JsValue instances or (supported) Python values
    :param drop_trailing_undefined: whether to drop trailing arguments that are undefined/None
    :return: the JsCode
    """
    args = [js_value(a) for a in args]
    last_index_to_include = len(args) - 1
    if drop_trailing_undefined:
        while last_index_to_include >= 0 and args[last_index_to_include].is_undefined():
            last_index_to_include -= 1
    args = args[:last_index_to_include+1]
    return JsCodeLiteral(", ".join(map(str, args)))


class JsObject(JsValue):
    def __init__(self):
        self.data = {}

    def add(self, key: str, value: Union[JsValue, PythonType]):
        self.data[key] = js_value(value)

    def add_string(self, key: str, value: str):
        self.data[key] = JsValue.string_value(value)

    def add_code_literal(self, key: str, value: str):
        self.data[key] = value

    def add_float(self, key: str, value: Union[float, int]):
        self.data[key] = JsValue.float_value(value)

    def add_json(self, key: str, value: Any):
        """
        :param key: key within the object
        :param value: any Python object which can be converted to JSON
        """
        self.add_code_literal(key, json.dumps(value))

    def get_js_code(self):
        return "{" + ", ".join(f'"{k}": {v}' for k, v in self.data.items()) + "}"

    def __len__(self):
        return len(self.data)


class JsClassInstance(JsValueLiteral):
    def __init__(self, class_name, *args: Union[JsValue, PythonType]):
        arg_list = js_arg_list(*args, drop_trailing_undefined=False)
        super().__init__(f"new {class_name}({arg_list})")


class JsList(JsValueLiteral):
    def __init__(self, *values: Union[JsValue, PythonType]):
        super().__init__(list_string([js_value(x) for x in values]))
