import pickle
import hashlib
from typing import Any


def pickle_hash(o: Any, algorithm='sha1', with_class_name=False) -> str:
    """
    Computes a deterministic hash code based on the contents of the given object

    :param o: any picklable object
    :param algorithm: the name of a hashing algorithm supported by hashlib
    :param with_class_name: if True, prepend o's class name to the returned hex string
    :return: the object's hash code as a hex string
    """
    s = pickle.dumps(o)
    result = hashlib.new(algorithm, s).hexdigest()
    if with_class_name:
        result = o.__class__.__name__ + result
    return result


def str_hash(o: Any, algorithm='sha1', with_class_name=False) -> str:
    """
    Computes a deterministic hash code based on the string representation of the given object

    :param o: any object
    :param algorithm: the name of a hashing algorithm supported by hashlib
    :param with_class_name: if True, prepend o's class name to the returned hex string
    :return: the object's hash code as a hex string
    """
    s = str(o)
    result = hashlib.new(algorithm, s).hexdigest()
    if with_class_name:
        result = o.__class__.__name__ + result
    return result