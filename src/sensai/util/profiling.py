import functools
import logging
from typing import Optional, Callable
from time import time


log = logging.getLogger(__name__)


def profiled(func: Optional[Callable] = None, *, sampling_interval_seconds: Optional[float] = None,
        print_to_console=True, log_to_file=False, file_name: Optional[str] = None):
    """
    Function decorator for profiling the annotated/given function with the pyinstrument profiler
    :param func: the function to be profiled
    :param sampling_interval_seconds: sampling every <sampling_interval_seconds> seconds, if None use default parameter of pyintrument profiler
    :param print_to_console: if, true print profiler output to console
    :param log_to_file: if true, logs profiler output to html file
    :param file_name: optional file name for log, if None defaults to file name derived from function name
    :return: the wrapped function
    """
    from pyinstrument import Profiler

    def wrapper(_func):
        @functools.wraps(_func)
        def _wrapper(*args, **kwargs):
            _profiler = Profiler(sampling_interval_seconds) if sampling_interval_seconds is not None else Profiler()
            _profiler.start()
            result = _func(*args, **kwargs)
            _profiler.stop()
            if print_to_console:
                print(_profiler.output_text(unicode=True, color=True))
            if log_to_file:
                name = file_name if file_name is not None else _func.__name__ + ".profile"
                with open(f"{name}.html", "w") as f:
                    f.write(_profiler.output_html())
            return result
        return _wrapper

    if func:
        return wrapper(func)

    return wrapper


def timed(func):
    """
    Function decorator which logs execution times of the wrapped/annotated function at INFO level
    :param func: the function whose execution time to log
    :return: the wrapped function
    """
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            log.info(f"Execution time of {func.__qualname__}: {end_ if end_ > 0 else 0} ms")
    return _wrapper