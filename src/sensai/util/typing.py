from typing import NamedTuple, Union


class PandasNamedTuple(NamedTuple):
    """
    This class is used for type annotations only
    """
    Index: Union[int, str]
