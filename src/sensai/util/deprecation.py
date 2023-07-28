import warnings
import logging


log = logging.getLogger(__name__)


def deprecated(message):
    """
    This is a decorator for functions to mark them as deprecated, issuing a warning when the function is called

    :param message: message, which can indicate the new recommended approach or reason for deprecation
    :return: decorated function
    """
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            msg = "{} is a deprecated function. {}".format(func.__name__, message)
            if logging.Logger.root.hasHandlers():
                log.warning(msg)
            else:
                warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator
