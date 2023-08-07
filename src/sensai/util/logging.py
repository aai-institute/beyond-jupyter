import atexit
import logging as lg
import sys
import time
from datetime import datetime
from io import StringIO
from logging import *
from typing import List, Callable, Any, Optional

import pandas as pd

log = getLogger(__name__)

LOG_DEFAULT_FORMAT = '%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s'
_logFormat = LOG_DEFAULT_FORMAT


def remove_log_handlers():
    """
    Removes all current log handlers
    """
    logger = getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


def is_log_handler_active(handler):
    return handler in getLogger().handlers


# noinspection PyShadowingBuiltins
def configure(format=LOG_DEFAULT_FORMAT, level=lg.DEBUG):
    """
    Configures logging to stdout with the given format and log level,
    also configuring the default log levels of some overly verbose libraries as well as some pandas output options.

    :param format: the log format
    :param level: the minimum log level
    """
    global _logFormat
    _logFormat = format
    remove_log_handlers()
    basicConfig(level=level, format=format, stream=sys.stdout)
    getLogger("matplotlib").setLevel(lg.INFO)
    getLogger("urllib3").setLevel(lg.INFO)
    getLogger("msal").setLevel(lg.INFO)
    pd.set_option('display.max_colwidth', 255)


# noinspection PyShadowingBuiltins
def run_main(main_fn: Callable[[], Any], format=LOG_DEFAULT_FORMAT, level=lg.DEBUG):
    """
    Configures logging with the given parameters, ensuring that any exceptions that occur during
    the execution of the given function are logged.
    Logs two additional messages, one before the execution of the function, and one upon its completion.

    :param main_fn: the function to be executed
    :param format: the log message format
    :param level: the minimum log level
    :return: the result of `main_fn`
    """
    configure(format=format, level=level)
    log.info("Starting")
    try:
        result = main_fn()
        log.info("Done")
        return result
    except Exception as e:
        log.error("Exception during script execution", exc_info=e)


def datetime_tag() -> str:
    """
    :return: a string tag for use in log file names which contains the current date and time (compact but readable)
    """
    return datetime.now().strftime('%Y%m%d-%H%M%S')


_fileLoggerPaths: List[str] = []
_isAtExitReportFileLoggerRegistered = False
_memoryLogStream: Optional[StringIO] = None


def _at_exit_report_file_logger():
    for path in _fileLoggerPaths:
        print(f"A log file was saved to {path}")


def add_file_logger(path):
    global _isAtExitReportFileLoggerRegistered
    log.info(f"Logging to {path} ...")
    handler = FileHandler(path)
    handler.setFormatter(Formatter(_logFormat))
    Logger.root.addHandler(handler)
    _fileLoggerPaths.append(path)
    if not _isAtExitReportFileLoggerRegistered:
        atexit.register(_at_exit_report_file_logger)
        _isAtExitReportFileLoggerRegistered = True


def add_memory_logger() -> None:
    """
    Enables in-memory logging (if it is not already enabled), i.e. all log statements are written to a memory buffer and can later be
    read via function `get_memory_log()`
    """
    global _memoryLogStream
    if _memoryLogStream is not None:
        return
    _memoryLogStream = StringIO()
    handler = StreamHandler(_memoryLogStream)
    handler.setFormatter(Formatter(_logFormat))
    Logger.root.addHandler(handler)


def get_memory_log():
    return _memoryLogStream.getvalue()


class StopWatch:
    """
    Represents a stop watch for timing an execution. Constructing an instance starts the stopwatch.
    """
    def __init__(self, start=True):
        self.start_time = time.time()
        self._elapsed_secs = 0.0
        self.is_running = start

    def reset(self, start=True):
        """
        Resets the stopwatch, setting the elapsed time to zero.

        :param start: whether to start the stopwatch immediately
        """
        self.start_time = time.time()
        self._elapsed_secs = 0.0
        self.is_running = start

    def restart(self):
        """
        Resets the stopwatch (setting the elapsed time to zero) and restarts it immediately
        """
        self.reset(start=True)

    def _get_elapsed_time_since_last_start(self):
        if self.is_running:
            return time.time() - self.start_time
        else:
            return 0

    def pause(self):
        """
        Pauses the stopwatch. It can be resumed via method 'resume'.
        """
        self._elapsed_secs += self._get_elapsed_time_since_last_start()
        self.is_running = False

    def resume(self):
        """
        Resumes the stopwatch (assuming that it is currently paused). If the stopwatch is not paused,
        the method has no effect (and a warning is logged).
        """
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
        else:
            log.warning("Stopwatch is already running (resume has not effect)")

    def get_elapsed_time_secs(self) -> float:
        """
        Gets the total elapsed time, in seconds, on this stopwatch.

        :return: the elapsed time in seconds
        """
        return self._elapsed_secs + self._get_elapsed_time_since_last_start()

    def get_elapsed_timedelta(self) -> pd.Timedelta:
        """
        :return: the elapsed time as a pandas.Timedelta object
        """
        return pd.Timedelta(self.get_elapsed_time_secs(), unit="s")

    def get_elapsed_time_string(self) -> str:
        """
        :return: a string representation of the elapsed time
        """
        secs = self.get_elapsed_time_secs()
        if secs < 60:
            return f"{secs:.3f} seconds"
        else:
            return str(pd.Timedelta(secs, unit="s"))


class StopWatchManager:
    """
    A singleton which manages a pool of named stopwatches, such that executions to be timed by referring to a name only -
    without the need for a limited scope.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = StopWatchManager(42)
        return cls._instance

    def __init__(self, secret):
        if secret != 42:
            raise Exception("Use only the singleton instance via getInstance")
        self._stopWatches = {}

    def start(self, name):
        self._stopWatches[name] = time.time()

    def stop(self, name) -> float:
        """
        :param name: the name of the stopwatch
        :return: the time that has passed in seconds
        """
        time_passed_secs = time.time() - self._stopWatches[name]
        del self._stopWatches[name]
        return time_passed_secs

    def is_running(self, name):
        return name in self._stopWatches


class LogTime:
    """
    An execution time logger which can be conveniently applied using a with-statement - in order to log the executing time of the respective
    with-block.
    """

    def __init__(self, name, enabled=True, logger: Logger = None):
        """
        :param name: the name of the event whose time is to be logged upon completion as "<name> completed in <time>"
        :param enabled: whether the logging is actually enabled; can be set to False to disable logging without necessitating
            changes to client code
        :param logger: the logger to use; if None, use the logger of LogTime's module
        """
        self.name = name
        self.enabled = enabled
        self.stopwatch = None
        self.logger = logger if logger is not None else log

    def start(self):
        """
        Starts the stopwatch
        """
        self.stopwatch = StopWatch()
        if self.enabled:
            self.logger.info(f"{self.name} starting ...")

    def stop(self):
        """
        Stops the stopwatch and logs the time taken (if enabled)
        """
        if self.stopwatch is not None and self.enabled:
            self.logger.info(f"{self.name} completed in {self.stopwatch.get_elapsed_time_string()}")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __enter__(self):
        self.start()
        return self
