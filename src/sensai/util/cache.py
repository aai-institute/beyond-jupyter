import atexit
import enum
import glob
import logging
import os
import pickle
import re
import sqlite3
import threading
import time
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, TypeVar, Generic, Union

from .hash import pickle_hash
from .pickle import load_pickle, dump_pickle, setstate

log = logging.getLogger(__name__)

T = TypeVar("T")
TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
TData = TypeVar("TData")


class BoxedValue(Generic[TValue]):
    """
    Container for a value, which can be used in caches where values may be None (to differentiate the value not being present in the cache
    from the cached value being None)
    """
    def __init__(self, value: TValue):
        self.value = value


class PersistentKeyValueCache(Generic[TKey, TValue], ABC):
    @abstractmethod
    def set(self, key: TKey, value: TValue):
        """
        Sets a cached value

        :param key: the key under which to store the value
        :param value: the value to store; since None is used indicate the absence of a value, None should not be
            used a value
        """
        pass

    @abstractmethod
    def get(self, key: TKey) -> Optional[TValue]:
        """
        Retrieves a cached value

        :param key: the lookup key
        :return: the cached value or None if no value is found
        """
        pass


class PersistentList(Generic[TValue], ABC):
    @abstractmethod
    def append(self, item: TValue):
        """
        Adds an item to the cache

        :param item: the item to store
        """
        pass

    @abstractmethod
    def iter_items(self) -> Iterator[TValue]:
        """
        Iterates over the items in the persisted list

        :return: generator of item
        """
        pass


class DelayedUpdateHook:
    """
    Ensures that a given function is executed after an update happens, but delay the execution until
    there are no further updates for a certain time period
    """
    def __init__(self, fn: Callable[[], Any], time_period_secs, periodically_executed_fn: Optional[Callable[[], Any]] = None):
        """
        :param fn: the function to eventually call after an update
        :param time_period_secs: the time that must pass while not receiving further updates for fn to be called
        :param periodically_executed_fn: a function to execute periodically (every timePeriodSecs seconds) in the busy waiting loop,
            which may, for example, log information or apply additional executions, which must not interfere with the correctness of
            the execution of fn
        """
        self.periodicallyExecutedFn = periodically_executed_fn
        self.fn = fn
        self.timePeriodSecs = time_period_secs
        self._lastUpdateTime = None
        self._thread = None
        self._threadLock = threading.Lock()

    def handle_update(self):
        """
        Notifies of an update and ensures that the function passed at construction is eventually called
        (after no more updates are received within the respective time window)
        """
        self._lastUpdateTime = time.time()

        def do_periodic_check():
            while True:
                time.sleep(self.timePeriodSecs)
                time_passed_since_last_update = time.time() - self._lastUpdateTime
                if self.periodicallyExecutedFn is not None:
                    self.periodicallyExecutedFn()
                if time_passed_since_last_update >= self.timePeriodSecs:
                    self.fn()
                    return

        # noinspection DuplicatedCode
        if self._thread is None or not self._thread.is_alive():
            self._threadLock.acquire()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=do_periodic_check, daemon=False)
                self._thread.start()
            self._threadLock.release()


class PeriodicUpdateHook:
    """
    Periodically checks whether a function shall be called as a result of an update, the function potentially
    being non-atomic (i.e. it may take a long time to execute such that new updates may come in while it is
    executing). Two function all mechanisms are in place:

        * a function which is called if there has not been a new update for a certain time period (which may be called
          several times if updates come in while the function is being executed)
        * a function which is called periodically

    """
    def __init__(self, check_interval_secs: float, no_update_time_period_secs: float = None, no_update_fn: Callable[[], Any] = None,
            periodic_fn: Optional[Callable[[], Any]] = None):
        """
        :param check_interval_secs: the time period, in seconds, between checks
        :param no_update_time_period_secs: the time period after which to execute noUpdateFn if no further updates have come in.
            This must be at least as large as checkIntervalSecs. If None, use checkIntervalSecs.
        :param no_update_fn: the function to call if there have been no further updates for noUpdateTimePeriodSecs seconds
        :param periodic_fn: a function to execute periodically (every checkIntervalSecs seconds) in the busy waiting loop,
            which may, for example, log information or apply additional executions, which must not interfere with the correctness of
            the execution of fn
        """
        if no_update_time_period_secs is None:
            no_update_time_period_secs = check_interval_secs
        elif no_update_time_period_secs < check_interval_secs:
            raise ValueError("noUpdateTimePeriodSecs must be at least as large as checkIntervalSecs")
        self._periodic_fn = periodic_fn
        self._check_interval_secs = check_interval_secs
        self._no_update_time_period_secs = no_update_time_period_secs
        self._no_update_fn = no_update_fn
        self._last_update_time = None
        self._thread = None
        self._thread_lock = threading.Lock()

    def handle_update(self):
        """
        Notifies of an update, making sure the functions given at construction will be called as specified
        """
        self._last_update_time = time.time()

        def do_periodic_check():
            while True:
                time.sleep(self._check_interval_secs)
                check_time = time.time()
                if self._periodic_fn is not None:
                    self._periodic_fn()
                time_passed_since_last_update = check_time - self._last_update_time
                if time_passed_since_last_update >= self._no_update_time_period_secs:
                    if self._no_update_fn is not None:
                        self._no_update_fn()
                    # if no further updates have come in, we terminate the thread
                    if self._last_update_time < check_time:
                        return

        # noinspection DuplicatedCode
        if self._thread is None or not self._thread.is_alive():
            self._thread_lock.acquire()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=do_periodic_check, daemon=False)
                self._thread.start()
            self._thread_lock.release()


class PicklePersistentKeyValueCache(PersistentKeyValueCache[TKey, TValue]):
    """
    Represents a key-value cache as a dictionary which is persisted in a file using pickle
    """
    def __init__(self, pickle_path, version=1, save_on_update=True, deferred_save_delay_secs=1.0):
        """
        :param pickle_path: the path of the file where the cache values are to be persisted
        :param version: the version of cache entries. If a persisted cache with a non-matching version is found,
            it is discarded
        :param save_on_update: whether to persist the cache after an update; the cache is saved in a deferred
            manner and will be saved after deferredSaveDelaySecs if no new updates have arrived in the meantime,
            i.e. it will ultimately be saved deferredSaveDelaySecs after the latest update
        :param deferred_save_delay_secs: the number of seconds to wait for additional data to be added to the cache
            before actually storing the cache after a cache update
        """
        self.deferred_save_delay_secs = deferred_save_delay_secs
        self.pickle_path = pickle_path
        self.version = version
        self.save_on_update = save_on_update
        cache_found = False
        if os.path.exists(pickle_path):
            try:
                log.info(f"Loading cache from {pickle_path}")
                persisted_version, self.cache = load_pickle(pickle_path)
                if persisted_version == version:
                    cache_found = True
            except EOFError:
                log.warning(f"The cache file in {pickle_path} is corrupt")
        if not cache_found:
            self.cache = {}
        self._update_hook = DelayedUpdateHook(self.save, deferred_save_delay_secs)
        self._write_lock = threading.Lock()

    def save(self):
        """
        Saves the cache in the file whose path was provided at construction
        """
        with self._write_lock:  # avoid concurrent modification while saving
            log.info(f"Saving cache to {self.pickle_path}")
            dump_pickle((self.version, self.cache), self.pickle_path)

    def get(self, key: TKey) -> Optional[TValue]:
        return self.cache.get(key)

    def set(self, key: TKey, value: TValue):
        with self._write_lock:
            self.cache[key] = value
            if self.save_on_update:
                self._update_hook.handle_update()


class SlicedPicklePersistentList(PersistentList):
    """
    Object handling the creation and access to sliced pickle caches
    """
    def __init__(self, directory, pickle_base_name, num_entries_per_slice=100000):
        """
        :param directory: path to the directory where the sliced caches are to be stored
        :param pickle_base_name: base name for the pickle, where slices will have the names {pickleBaseName}_sliceX.pickle
        :param num_entries_per_slice: how many entries should be stored in each cache
        """
        self.directory = directory
        self.pickleBaseName = pickle_base_name
        self.numEntriesPerSlice = num_entries_per_slice

        # Set up the variables for the sliced cache
        self.slice_id = 0
        self.index_in_slice = 0
        self.cache_of_slice = []

        # Search directory for already present sliced caches
        self.slicedFiles = self._find_sliced_caches()

        # Helper variable to ensure object is only modified within a with-clause
        self._currentlyInWithClause = False

    def __enter__(self):
        self._currentlyInWithClause = True
        if self.cache_exists():
            # Reset state to enable the appending of more items to the cache
            self._set_last_cache_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dump()
        self._currentlyInWithClause = False

    def append(self, item):
        """
        Append item to cache
        :param item: entry in the cache
        """
        if not self._currentlyInWithClause:
            raise Exception("Class needs to be instantiated within a with-clause to ensure correct storage")

        if (self.index_in_slice + 1) % self.numEntriesPerSlice == 0:
            self._dump()

        self.cache_of_slice.append(item)
        self.index_in_slice += 1

    def iter_items(self) -> Iterator[Any]:
        """
        Iterate over entries in the sliced cache
        :return: iterator over all items in the cache
        """
        for filePath in self.slicedFiles:
            log.info(f"Loading sliced pickle list from {filePath}")
            cached_pickle = self._load_pickle(filePath)
            for item in cached_pickle:
                yield item

    def clear(self):
        """
        Clears the cache if it exists
        """
        if self.cache_exists():
            for filePath in self.slicedFiles:
                os.unlink(filePath)

    def cache_exists(self) -> bool:
        """
        Does this cache already exist
        :return: True if cache exists, False if not
        """
        return len(self.slicedFiles) > 0

    def _set_last_cache_state(self):
        """
        Sets the state so as to be able to add items to an existing cache
        """
        log.info("Resetting last state of cache...")
        self.slice_id = len(self.slicedFiles) - 1
        self.cache_of_slice = self._load_pickle(self._pickle_path(self.slice_id))
        self.index_in_slice = len(self.cache_of_slice) - 1
        if self.index_in_slice >= self.numEntriesPerSlice:
            self._next_slice()

    def _dump(self):
        """
        Dumps the current cache (if non-empty)
        """
        if len(self.cache_of_slice) > 0:
            pickle_path = self._pickle_path(str(self.slice_id))
            log.info(f"Saving sliced cache to {pickle_path}")
            dump_pickle(self.cache_of_slice, pickle_path)
            self.slicedFiles.append(pickle_path)

            # Update slice number and reset indexing and cache
            self._next_slice()
        else:
            log.warning("Unexpected behavior: Dump was called when cache of slice is 0!")

    def _next_slice(self):
        """
        Updates sliced cache state for the next slice
        """
        self.slice_id += 1
        self.index_in_slice = 0
        self.cache_of_slice = []

    def _find_sliced_caches(self) -> List[str]:
        """
        Finds all pickled slices associated with this cache
        :return: list of sliced pickled files
        """
        # glob.glob permits the usage of unix-style pathnames matching. (below we find all ..._slice*.pickle files)
        list_of_file_names = glob.glob(self._pickle_path("*"))
        # Sort the slices to ensure it is in the same order as they was produced (regex replaces everything not a number with empty string).
        list_of_file_names.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        return list_of_file_names

    def _load_pickle(self, pickle_path: str) -> List[Any]:
        """
        Loads pickle if file path exists, and persisted version is correct.
        :param pickle_path: file path
        :return: list with objects
        """
        cached_pickle = []
        if os.path.exists(pickle_path):
            try:
                cached_pickle = load_pickle(pickle_path)
            except EOFError:
                log.warning(f"The cache file in {pickle_path} is corrupt")
        else:
            raise Exception(f"The file {pickle_path} does not exist!")
        return cached_pickle

    def _pickle_path(self, slice_suffix) -> str:
        return f"{os.path.join(self.directory, self.pickleBaseName)}_slice{slice_suffix}.pickle"


class SqliteConnectionManager:
    _connections: List[sqlite3.Connection] = []
    _atexit_handler_registered = False

    @classmethod
    def _register_at_exit_handler(cls):
        if not cls._atexit_handler_registered:
            cls._atexit_handler_registered = True
            atexit.register(cls._cleanup)

    @classmethod
    def open_connection(cls, path):
        cls._register_at_exit_handler()
        conn = sqlite3.connect(path, check_same_thread=False)
        cls._connections.append(conn)
        return conn

    @classmethod
    def _cleanup(cls):
        for conn in cls._connections:
            conn.close()
        cls._connections = []


class SqlitePersistentKeyValueCache(PersistentKeyValueCache[TKey, TValue]):
    class KeyType(enum.Enum):
        STRING = ("VARCHAR(%d)", )
        INTEGER = ("LONG", )

    def __init__(self, path, table_name="cache", deferred_commit_delay_secs=1.0, key_type: KeyType = KeyType.STRING,
            max_key_length=255):
        """
        :param path: the path to the file that is to hold the SQLite database
        :param table_name: the name of the table to create in the database
        :param deferred_commit_delay_secs: the time frame during which no new data must be added for a pending transaction to be committed
        :param key_type: the type to use for keys; for complex keys (i.e. tuples), use STRING (conversions to string are automatic)
        :param max_key_length: the maximum key length for the case where the key_type can be parametrised (e.g. STRING)
        """
        self.path = path
        self.conn = SqliteConnectionManager.open_connection(path)
        self.table_name = table_name
        self.max_key_length = 255
        self.key_type = key_type
        self._update_hook = DelayedUpdateHook(self._commit, deferred_commit_delay_secs)
        self._num_entries_to_be_committed = 0
        self._conn_mutex = threading.Lock()

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table';")
        if table_name not in [r[0] for r in cursor.fetchall()]:
            log.info(f"Creating cache table '{self.table_name}' in {path}")
            key_db_type = key_type.value[0]
            if "%d" in key_db_type:
                key_db_type = key_db_type % max_key_length
            cursor.execute(f"CREATE TABLE {table_name} (cache_key {key_db_type} PRIMARY KEY, cache_value BLOB);")
        cursor.close()

    def _key_db_value(self, key):
        if self.key_type == self.KeyType.STRING:
            s = str(key)
            if len(s) > self.max_key_length:
                raise ValueError(f"Key too long, maximal key length is {self.max_key_length}")
            return s
        elif self.key_type == self.KeyType.INTEGER:
            return int(key)
        else:
            raise Exception(f"Unhandled key type {self.key_type}")

    def _commit(self):
        self._conn_mutex.acquire()
        try:
            log.info(f"Committing {self._num_entries_to_be_committed} cache entries to the SQLite database {self.path}")
            self.conn.commit()
            self._num_entries_to_be_committed = 0
        finally:
            self._conn_mutex.release()

    def set(self, key: TKey, value: TValue):
        self._conn_mutex.acquire()
        try:
            cursor = self.conn.cursor()
            key = self._key_db_value(key)
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE cache_key=?", (key,))
            if cursor.fetchone()[0] == 0:
                cursor.execute(f"INSERT INTO {self.table_name} (cache_key, cache_value) VALUES (?, ?)",
                               (key, pickle.dumps(value)))
            else:
                cursor.execute(f"UPDATE {self.table_name} SET cache_value=? WHERE cache_key=?", (pickle.dumps(value), key))
            self._num_entries_to_be_committed += 1
            cursor.close()
        finally:
            self._conn_mutex.release()

        self._update_hook.handle_update()

    def _execute(self, cursor, *query):
        try:
            cursor.execute(*query)
        except sqlite3.DatabaseError as e:
            raise Exception(f"Error executing query for {self.path}: {e}")

    def get(self, key: TKey) -> Optional[TValue]:
        self._conn_mutex.acquire()
        try:
            cursor = self.conn.cursor()
            key = self._key_db_value(key)
            self._execute(cursor, f"SELECT cache_value FROM {self.table_name} WHERE cache_key=?", (key,))
            row = cursor.fetchone()
            cursor.close()
            if row is None:
                return None
            return pickle.loads(row[0])
        finally:
            self._conn_mutex.release()

    def __len__(self):
        self._conn_mutex.acquire()
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            cnt = cursor.fetchone()[0]
            cursor.close()
            return cnt
        finally:
            self._conn_mutex.release()

    def iter_items(self):
        self._conn_mutex.acquire()
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT cache_key, cache_value FROM {self.table_name}")
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                yield row[0], pickle.loads(row[1])
            cursor.close()
        finally:
            self._conn_mutex.release()


class SqlitePersistentList(PersistentList):
    def __init__(self, path):
        self.keyValueCache = SqlitePersistentKeyValueCache(path, key_type=SqlitePersistentKeyValueCache.KeyType.INTEGER)
        self.nextKey = len(self.keyValueCache)

    def append(self, item):
        self.keyValueCache.set(self.nextKey, item)
        self.nextKey += 1

    def iter_items(self):
        for item in self.keyValueCache.iter_items():
            yield item[1]


class CachedValueProviderMixin(Generic[TKey, TValue, TData], ABC):
    """
    Represents a value provider that can provide values associated with (hashable) keys via a cache or, if
    cached values are not yet present, by computing them.
    """
    def __init__(self, cache: Optional[PersistentKeyValueCache[TKey, TValue]] = None,
            cache_factory: Optional[Callable[[], PersistentKeyValueCache[TKey, TValue]]] = None, persist_cache=False, box_values=False):
        """
        :param cache: the cache to use or None. If None, caching will be disabled
        :param cache_factory: a factory with which to create the cache (or recreate it after unpickling if `persistCache` is False, in which
            case this factory must be picklable)
        :param persist_cache: whether to persist the cache when pickling
        :param box_values: whether to box values, such that None is admissible as a value
        """
        self._persistCache = persist_cache
        self._boxValues = box_values
        self._cache = cache
        self._cacheFactory = cache_factory
        if self._cache is None and cache_factory is not None:
            self._cache = cache_factory()

    def __getstate__(self):
        if not self._persistCache:
            d = self.__dict__.copy()
            d["_cache"] = None
            return d
        return self.__dict__

    def __setstate__(self, state):
        setstate(CachedValueProviderMixin, self, state, renamed_properties={"persistCache": "_persistCache"})
        if not self._persistCache and self._cacheFactory is not None:
            self._cache = self._cacheFactory()

    def _provide_value(self, key, data: Optional[TData] = None):
        """
        Provides the value for the key by retrieving the associated value from the cache or, if no entry in the
        cache is found, by computing the value via _computeValue

        :param key: the key for which to provide the value
        :param data: optional data required to compute a value
        :return: the retrieved or computed value
        """
        if self._cache is None:
            return self._compute_value(key, data)
        value = self._cache.get(key)
        if value is None:
            value = self._compute_value(key, data)
            self._cache.set(key, value if not self._boxValues else BoxedValue(value))
        else:
            if self._boxValues:
                value: BoxedValue[TValue]
                value = value.value
        return value

    @abstractmethod
    def _compute_value(self, key: TKey, data: Optional[TData]) -> TValue:
        """
        Computes the value for the given key

        :param key: the key for which to compute the value
        :return: the computed value
        """
        pass


def cached(fn: Callable[[], T], pickle_path, function_name=None, validity_check_fn: Optional[Callable[[T], bool]] = None,
        backend="pickle", protocol=pickle.HIGHEST_PROTOCOL, load=True, version=None) -> T:
    """
    :param fn: the function whose result is to be cached
    :param pickle_path: the path in which to store the cached result
    :param function_name: the name of the function fn (for the case where its __name__ attribute is not
        informative)
    :param validity_check_fn: an optional function to call in order to check whether a cached result is still valid;
        the function shall return True if the result is still valid and false otherwise. If a cached result is invalid,
        the function fn is called to compute the result and the cached result is updated.
    :param backend: pickle or joblib
    :param protocol: the pickle protocol version
    :param load: whether to load a previously persisted result; if False, do not load an old result but store the newly computed result
    :param version: if not None, previously persisted data will only be returned if it was stored with the same version
    :return: the result (either obtained from the cache or the function)
    """
    if function_name is None:
        function_name = fn.__name__

    def call_fn_and_cache_result():
        res = fn()
        log.info(f"Saving cached result in {pickle_path}")
        if version is not None:
            persisted_res = {"__cacheVersion": version, "obj": res}
        else:
            persisted_res = res
        dump_pickle(persisted_res, pickle_path, backend=backend, protocol=protocol)
        return res

    if os.path.exists(pickle_path):
        if load:
            log.info(f"Loading cached result of function '{function_name}' from {pickle_path}")
            result = load_pickle(pickle_path, backend=backend)
            if validity_check_fn is not None:
                if not validity_check_fn(result):
                    log.info(f"Cached result is no longer valid, recomputing ...")
                    result = call_fn_and_cache_result()
            if version is not None:
                cached_version = None
                if type(result) == dict:
                    cached_version = result.get("__cacheVersion")
                if cached_version != version:
                    log.info(f"Cached result has incorrect version ({cached_version}, expected {version}), recomputing ...")
                    result = call_fn_and_cache_result()
                else:
                    result = result["obj"]
            return result
        else:
            log.info(f"Ignoring previously stored result in {pickle_path}, calling function '{function_name}' ...")
            return call_fn_and_cache_result()
    else:
        log.info(f"No cached result found in {pickle_path}, calling function '{function_name}' ...")
        return call_fn_and_cache_result()


# TODO consider renaming to pickle_cached (in line with other decorators)
class PickleCached(object):
    """
    Function decorator for caching function results via pickle
    """
    def __init__(self, cache_base_path: str, filename_prefix: str = None, filename: str = None, backend="pickle",
            protocol=pickle.HIGHEST_PROTOCOL, load=True, version=None):
        """
        :param cache_base_path: the directory where the pickle cache file will be stored
        :param filename_prefix: a prefix of the name of the cache file to be created, to which the function name and, where applicable,
            a hash code of the function arguments will be appended and ".cache.pickle" will be appended; if None, use "" (if filename
            has not been provided)
        :param filename: the full file name of the cache file to be created; if the function takes arguments, the filename must
            contain a placeholder '%s' for the argument hash
        :param backend: the serialisation backend to use (see dumpPickle)
        :param protocol: the pickle protocol version to use
        :param load: whether to load a previously persisted result; if False, do not load an old result but store the newly computed result
        :param version: if not None, previously persisted data will only be returned if it was stored with the same version
        """
        self.filename = filename
        self.cache_base_path = cache_base_path
        self.filename_prefix = filename_prefix
        self.backend = backend
        self.protocol = protocol
        self.load = load
        self.version = version

        if self.filename_prefix is None:
            self.filename_prefix = ""
        else:
            self.filename_prefix += "-"

    def __call__(self, fn: Callable, *_args, **_kwargs):

        def wrapped(*args, **kwargs):
            hash_code_str = None
            have_args = len(args) > 0 or len(kwargs) > 0
            if have_args:
                hash_code_str = pickle_hash((args, kwargs))
            if self.filename is None:
                filename = self.filename_prefix + fn.__qualname__.replace(".<locals>.", ".")
                if hash_code_str is not None:
                    filename += "-" + hash_code_str
                filename += ".cache.pickle"
            else:
                if hash_code_str is not None:
                    if "%s" not in self.filename:
                        raise Exception("Function called with arguments but full cache filename contains no placeholder (%s) "
                                        "for argument hash")
                    filename = self.filename % hash_code_str
                else:
                    if "%s" in self.filename:
                        raise Exception("Function without arguments but full cache filename with placeholder (%s) was specified")
                    filename = self.filename
            pickle_path = os.path.join(self.cache_base_path, filename)
            return cached(lambda: fn(*args, **kwargs), pickle_path, function_name=fn.__name__, backend=self.backend, load=self.load,
                version=self.version)

        return wrapped


class LoadSaveInterface(ABC):
    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls: T, path: str) -> T:
        pass


class PickleLoadSaveMixin(LoadSaveInterface):
    def save(self, path: Union[str, Path], backend="pickle"):
        """
        Saves the instance as pickle

        :param path:
        :param backend: pickle or joblib
        """
        dump_pickle(self, path, backend=backend)

    @classmethod
    def load(cls, path: Union[str, Path], backend="pickle"):
        """
        Loads a class instance from pickle

        :param path:
        :param backend: pickle or joblib
        :return: instance of the present class
        """
        log.info(f"Loading instance of {cls} from {path}")
        result = load_pickle(path, backend=backend)
        if not isinstance(result, cls):
            raise Exception(f"Excepted instance of {cls}, instead got: {result.__class__.__name__}")
        return result
