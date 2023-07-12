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
from typing import Any, Callable, Iterator, List, Optional, TypeVar, Generic

from .hash import pickleHash
from .pickle import loadPickle, dumpPickle, setstate

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
    def iterItems(self) -> Iterator[TValue]:
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
    def __init__(self, fn: Callable[[], Any], timePeriodSecs, periodicallyExecutedFn: Optional[Callable[[], Any]] = None):
        """
        :param fn: the function to eventually call after an update
        :param timePeriodSecs: the time that must pass while not receiving further updates for fn to be called
        :param periodicallyExecutedFn: a function to execute periodically (every timePeriodSecs seconds) in the busy waiting loop,
            which may, for example, log information or apply additional executions, which must not interfere with the correctness of
            the execution of fn
        """
        self.periodicallyExecutedFn = periodicallyExecutedFn
        self.fn = fn
        self.timePeriodSecs = timePeriodSecs
        self._lastUpdateTime = None
        self._thread = None
        self._threadLock = threading.Lock()

    def handleUpdate(self):
        """
        Notifies of an update and ensures that the function passed at construction is eventually called
        (after no more updates are received within the respective time window)
        """
        self._lastUpdateTime = time.time()

        def doPeriodicCheck():
            while True:
                time.sleep(self.timePeriodSecs)
                timePassedSinceLastUpdate = time.time() - self._lastUpdateTime
                if self.periodicallyExecutedFn is not None:
                    self.periodicallyExecutedFn()
                if timePassedSinceLastUpdate >= self.timePeriodSecs:
                    self.fn()
                    return

        if self._thread is None or not self._thread.is_alive():
            self._threadLock.acquire()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=doPeriodicCheck, daemon=False)
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
    def __init__(self, checkIntervalSecs: float, noUpdateTimePeriodSecs: float = None, noUpdateFn: Callable[[], Any] = None,
            periodicFn: Optional[Callable[[], Any]] = None):
        """
        :param checkIntervalSecs: the time period, in seconds, between checks
        :param noUpdateTimePeriodSecs: the time period after which to execute noUpdateFn if no further updates have come in.
            This must be at least as large as checkIntervalSecs. If None, use checkIntervalSecs.
        :param noUpdateFn: the function to call if there have been no further updates for noUpdateTimePeriodSecs seconds
        :param periodicFn: a function to execute periodically (every checkIntervalSecs seconds) in the busy waiting loop,
            which may, for example, log information or apply additional executions, which must not interfere with the correctness of
            the execution of fn
        """
        if noUpdateTimePeriodSecs is None:
            noUpdateTimePeriodSecs = checkIntervalSecs
        elif noUpdateTimePeriodSecs < checkIntervalSecs:
            raise ValueError("noUpdateTimePeriodSecs must be at least as large as checkIntervalSecs")
        self._periodicFn = periodicFn
        self._checkIntervalSecs = checkIntervalSecs
        self._noUpdateTimePeriodSecs = noUpdateTimePeriodSecs
        self._noUpdateFn = noUpdateFn
        self._lastUpdateTime = None
        self._thread = None
        self._threadLock = threading.Lock()

    def handleUpdate(self):
        """
        Notifies of an update, making sure the functions given at construction will be called as specified
        """
        self._lastUpdateTime = time.time()

        def doPeriodicCheck():
            while True:
                time.sleep(self._checkIntervalSecs)
                checkTime = time.time()
                if self._periodicFn is not None:
                    self._periodicFn()
                timePassedSinceLastUpdate = checkTime - self._lastUpdateTime
                if timePassedSinceLastUpdate >= self._noUpdateTimePeriodSecs:
                    if self._noUpdateFn is not None:
                        self._noUpdateFn()
                    # if no further updates have come in, we terminate the thread
                    if self._lastUpdateTime < checkTime:
                        return

        if self._thread is None or not self._thread.is_alive():
            self._threadLock.acquire()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=doPeriodicCheck, daemon=False)
                self._thread.start()
            self._threadLock.release()


class PicklePersistentKeyValueCache(PersistentKeyValueCache[TKey, TValue]):
    """
    Represents a key-value cache as a dictionary which is persisted in a file using pickle
    """
    def __init__(self, picklePath, version=1, saveOnUpdate=True, deferredSaveDelaySecs=1.0):
        """
        :param picklePath: the path of the file where the cache values are to be persisted
        :param version: the version of cache entries. If a persisted cache with a non-matching version is found,
            it is discarded
        :param saveOnUpdate: whether to persist the cache after an update; the cache is saved in a deferred
            manner and will be saved after deferredSaveDelaySecs if no new updates have arrived in the meantime,
            i.e. it will ultimately be saved deferredSaveDelaySecs after the latest update
        :param deferredSaveDelaySecs: the number of seconds to wait for additional data to be added to the cache
            before actually storing the cache after a cache update
        """
        self.deferredSaveDelaySecs = deferredSaveDelaySecs
        self.picklePath = picklePath
        self.version = version
        self.saveOnUpdate = saveOnUpdate
        cacheFound = False
        if os.path.exists(picklePath):
            try:
                log.info(f"Loading cache from {picklePath}")
                persistedVersion, self.cache = loadPickle(picklePath)
                if persistedVersion == version:
                    cacheFound = True
            except EOFError:
                log.warning(f"The cache file in {picklePath} is corrupt")
        if not cacheFound:
            self.cache = {}
        self._updateHook = DelayedUpdateHook(self.save, deferredSaveDelaySecs)
        self._writeLock = threading.Lock()

    def save(self):
        """
        Saves the cache in the file whose path was provided at construction
        """
        with self._writeLock:  # avoid concurrent modification while saving
            log.info(f"Saving cache to {self.picklePath}")
            dumpPickle((self.version, self.cache), self.picklePath)

    def get(self, key: TKey) -> Optional[TValue]:
        return self.cache.get(key)

    def set(self, key: TKey, value: TValue):
        with self._writeLock:
            self.cache[key] = value
            if self.saveOnUpdate:
                self._updateHook.handleUpdate()


class SlicedPicklePersistentList(PersistentList):
    """
    Object handling the creation and access to sliced pickle caches
    """
    def __init__(self, directory, pickleBaseName, numEntriesPerSlice=100000):
        """
        :param directory: path to the directory where the sliced caches are to be stored
        :param pickleBaseName: base name for the pickle, where slices will have the names {pickleBaseName}_sliceX.pickle
        :param numEntriesPerSlice: how many entries should be stored in each cache
        """
        self.directory = directory
        self.pickleBaseName = pickleBaseName
        self.numEntriesPerSlice = numEntriesPerSlice

        # Set up the variables for the sliced cache
        self.sliceId = 0
        self.indexInSlice = 0
        self.cacheOfSlice = []

        # Search directory for already present sliced caches
        self.slicedFiles = self._findSlicedCaches()

        # Helper variable to ensure object is only modified within a with-clause
        self._currentlyInWithClause = False

    def __enter__(self):
        self._currentlyInWithClause = True
        if self.cacheExists():
            # Reset state to enable the appending of more items to the cache
            self._setLastCacheState()
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

        if (self.indexInSlice + 1) % self.numEntriesPerSlice == 0:
            self._dump()

        self.cacheOfSlice.append(item)
        self.indexInSlice += 1

    def iterItems(self) -> Iterator[Any]:
        """
        Iterate over entries in the sliced cache
        :return: iterator over all items in the cache
        """
        for filePath in self.slicedFiles:
            log.info(f"Loading sliced pickle list from {filePath}")
            cachedPickle = self._loadPickle(filePath)
            for item in cachedPickle:
                yield item

    def clear(self):
        """
        Clears the cache if it exists
        """
        if self.cacheExists():
            for filePath in self.slicedFiles:
                os.unlink(filePath)

    def cacheExists(self) -> bool:
        """
        Does this cache already exist
        :return: True if cache exists, False if not
        """
        return len(self.slicedFiles) > 0

    def _setLastCacheState(self):
        """
        Sets the state such as to be able to add items to an existant cache
        """
        log.info("Resetting last state of cache...")
        self.sliceId = len(self.slicedFiles) - 1
        self.cacheOfSlice = self._loadPickle(self._picklePath(self.sliceId))
        self.indexInSlice = len(self.cacheOfSlice) - 1
        if self.indexInSlice >= self.numEntriesPerSlice:
            self._nextSlice()

    def _dump(self):
        """
        Dumps the current cache (if non-empty)
        """
        if len(self.cacheOfSlice) > 0:
            picklePath = self._picklePath(str(self.sliceId))
            log.info(f"Saving sliced cache to {picklePath}")
            dumpPickle(self.cacheOfSlice, picklePath)
            self.slicedFiles.append(picklePath)

            # Update slice number and reset indexing and cache
            self._nextSlice()
        else:
            log.warning("Unexpected behavior: Dump was called when cache of slice is 0!")

    def _nextSlice(self):
        """
        Updates sliced cache state for the next slice
        """
        self.sliceId += 1
        self.indexInSlice = 0
        self.cacheOfSlice = []

    def _findSlicedCaches(self) -> List[str]:
        """
        Finds all pickled slices associated with this cache
        :return: list of sliced pickled files
        """
        # glob.glob permits the usage of unix-style pathnames matching. (below we find all ..._slice*.pickle files)
        listOfFileNames = glob.glob(self._picklePath("*"))
        # Sort the slices to ensure it is in the same order as they was produced (regex replaces everything not a number with empty string).
        listOfFileNames.sort(key=lambda f: int(re.sub('\D', '', f)))
        return listOfFileNames

    def _loadPickle(self, picklePath: str) -> List[Any]:
        """
        Loads pickle if file path exists, and persisted version is correct.
        :param picklePath: file path
        :return: list with objects
        """
        cachedPickle = []
        if os.path.exists(picklePath):
            try:
                cachedPickle = loadPickle(picklePath)
            except EOFError:
                log.warning(f"The cache file in {picklePath} is corrupt")
        else:
            raise Exception(f"The file {picklePath} does not exist!")
        return cachedPickle

    def _picklePath(self, sliceSuffix) -> str:
        return f"{os.path.join(self.directory, self.pickleBaseName)}_slice{sliceSuffix}.pickle"


class SqliteConnectionManager:
    _connections: List[sqlite3.Connection] = []
    _atexitHandlerRegistered = False

    @classmethod
    def _registerAtExitHandler(cls):
        if not cls._atexitHandlerRegistered:
            cls._atexitHandlerRegistered = True
            atexit.register(cls._cleanup)

    @classmethod
    def openConnection(cls, path):
        cls._registerAtExitHandler()
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

    def __init__(self, path, tableName="cache", deferredCommitDelaySecs=1.0, keyType: KeyType = KeyType.STRING,
            maxKeyLength=255):
        """
        :param path: the path to the file that is to hold the SQLite database
        :param tableName: the name of the table to create in the database
        :param deferredCommitDelaySecs: the time frame during which no new data must be added for a pending transaction to be committed
        :param keyType: the type to use for keys; for complex keys (i.e. tuples), use STRING (conversions to string are automatic)
        :param maxKeyLength: the maximum key length for the case where the keyType can be parametrised (e.g. STRING)
        """
        self.path = path
        self.conn = SqliteConnectionManager.openConnection(path)
        self.tableName = tableName
        self.maxKeyLength = 255
        self.keyType = keyType
        self._updateHook = DelayedUpdateHook(self._commit, deferredCommitDelaySecs)
        self._numEntriesToBeCommitted = 0
        self._connMutex = threading.Lock()

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table';")
        if tableName not in [r[0] for r in cursor.fetchall()]:
            log.info(f"Creating cache table '{self.tableName}' in {path}")
            keyDbType = keyType.value[0]
            if "%d" in keyDbType:
                keyDbType = keyDbType % maxKeyLength
            cursor.execute(f"CREATE TABLE {tableName} (cache_key {keyDbType} PRIMARY KEY, cache_value BLOB);")
        cursor.close()

    def _keyDbValue(self, key):
        if self.keyType == self.KeyType.STRING:
            s = str(key)
            if len(s) > self.maxKeyLength:
                raise ValueError(f"Key too long, maximal key length is {self.maxKeyLength}")
            return s
        elif self.keyType == self.KeyType.INTEGER:
            return int(key)
        else:
            raise Exception(f"Unhandled key type {self.keyType}")

    def _commit(self):
        self._connMutex.acquire()
        try:
            log.info(f"Committing {self._numEntriesToBeCommitted} cache entries to the SQLite database {self.path}")
            self.conn.commit()
            self._numEntriesToBeCommitted = 0
        finally:
            self._connMutex.release()

    def set(self, key: TKey, value: TValue):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            key = self._keyDbValue(key)
            cursor.execute(f"SELECT COUNT(*) FROM {self.tableName} WHERE cache_key=?", (key, ))
            if cursor.fetchone()[0] == 0:
                cursor.execute(f"INSERT INTO {self.tableName} (cache_key, cache_value) VALUES (?, ?)",
                               (key, pickle.dumps(value)))
            else:
                cursor.execute(f"UPDATE {self.tableName} SET cache_value=? WHERE cache_key=?", (pickle.dumps(value), key))
            self._numEntriesToBeCommitted += 1
            cursor.close()
        finally:
            self._connMutex.release()

        self._updateHook.handleUpdate()

    def _execute(self, cursor, *query):
        try:
            cursor.execute(*query)
        except sqlite3.DatabaseError as e:
            raise Exception(f"Error executing query for {self.path}: {e}")

    def get(self, key: TKey) -> Optional[TValue]:
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            key = self._keyDbValue(key)
            self._execute(cursor, f"SELECT cache_value FROM {self.tableName} WHERE cache_key=?", (key, ))
            row = cursor.fetchone()
            cursor.close()
            if row is None:
                return None
            return pickle.loads(row[0])
        finally:
            self._connMutex.release()

    def __len__(self):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.tableName}")
            cnt = cursor.fetchone()[0]
            cursor.close()
            return cnt
        finally:
            self._connMutex.release()

    def iterItems(self):
        self._connMutex.acquire()
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT cache_key, cache_value FROM {self.tableName}")
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                yield row[0], pickle.loads(row[1])
            cursor.close()
        finally:
            self._connMutex.release()


class SqlitePersistentList(PersistentList):
    def __init__(self, path):
        self.keyValueCache = SqlitePersistentKeyValueCache(path, keyType=SqlitePersistentKeyValueCache.KeyType.INTEGER)
        self.nextKey = len(self.keyValueCache)

    def append(self, item):
        self.keyValueCache.set(self.nextKey, item)
        self.nextKey += 1

    def iterItems(self):
        for item in self.keyValueCache.iterItems():
            yield item[1]


class CachedValueProviderMixin(Generic[TKey, TValue, TData], ABC):
    """
    Represents a value provider that can provide values associated with (hashable) keys via a cache or, if
    cached values are not yet present, by computing them.
    """
    def __init__(self, cache: Optional[PersistentKeyValueCache[TKey, TValue]] = None,
            cacheFactory: Optional[Callable[[], PersistentKeyValueCache[TKey, TValue]]] = None, persistCache=False, boxValues=False):
        """
        :param cache: the cache to use or None. If None, caching will be disabled
        :param cacheFactory: a factory with which to create the cache (or recreate it after unpickling if `persistCache` is False, in which
            case this factory must be picklable)
        :param persistCache: whether to persist the cache when pickling
        :param boxValues: whether to box values, such that None is admissible as a value
        """
        self._persistCache = persistCache
        self._boxValues = boxValues
        self._cache = cache
        self._cacheFactory = cacheFactory
        if self._cache is None and cacheFactory is not None:
            self._cache = cacheFactory()

    def __getstate__(self):
        if not self._persistCache:
            d = self.__dict__.copy()
            d["_cache"] = None
            return d
        return self.__dict__

    def __setstate__(self, state):
        setstate(CachedValueProviderMixin, self, state, renamedProperties={"persistCache": "_persistCache"})
        if not self._persistCache and self._cacheFactory is not None:
            self._cache = self._cacheFactory()

    def _provideValue(self, key, data: Optional[TData] = None):
        """
        Provides the value for the key by retrieving the associated value from the cache or, if no entry in the
        cache is found, by computing the value via _computeValue

        :param key: the key for which to provide the value
        :param data: optional data required to compute a value
        :return: the retrieved or computed value
        """
        if self._cache is None:
            return self._computeValue(key, data)
        value = self._cache.get(key)
        if value is None:
            value = self._computeValue(key, data)
            self._cache.set(key, value if not self._boxValues else BoxedValue(value))
        else:
            if self._boxValues:
                value: BoxedValue[TValue]
                value = value.value
        return value

    @abstractmethod
    def _computeValue(self, key: TKey, data: Optional[TData]) -> TValue:
        """
        Computes the value for the given key

        :param key: the key for which to compute the value
        :return: the computed value
        """
        pass


def cached(fn: Callable[[], T], picklePath, functionName=None, validityCheckFn: Optional[Callable[[T], bool]] = None,
        backend="pickle", protocol=pickle.HIGHEST_PROTOCOL, load=True, version=None) -> T:
    """
    :param fn: the function whose result is to be cached
    :param picklePath: the path in which to store the cached result
    :param functionName: the name of the function fn (for the case where its __name__ attribute is not
        informative)
    :param validityCheckFn: an optional function to call in order to check whether a cached result is still valid;
        the function shall return True if the result is still valid and false otherwise. If a cached result is invalid,
        the function fn is called to compute the result and the cached result is updated.
    :param backend: pickle or joblib
    :param protocol: the pickle protocol version
    :param load: whether to load a previously persisted result; if False, do not load an old result but store the newly computed result
    :param version: if not None, previously persisted data will only be returned if it was stored with the same version
    :return: the result (either obtained from the cache or the function)
    """
    if functionName is None:
        functionName = fn.__name__

    def callFnAndCacheResult():
        res = fn()
        log.info(f"Saving cached result in {picklePath}")
        if version is not None:
            persistedRes = {"__cacheVersion": version, "obj": res}
        else:
            persistedRes = res
        dumpPickle(persistedRes, picklePath, backend=backend, protocol=protocol)
        return res

    if os.path.exists(picklePath):
        if load:
            log.info(f"Loading cached result of function '{functionName}' from {picklePath}")
            result = loadPickle(picklePath, backend=backend)
            if validityCheckFn is not None:
                if not validityCheckFn(result):
                    log.info(f"Cached result is no longer valid, recomputing ...")
                    result = callFnAndCacheResult()
            if version is not None:
                cachedVersion = None
                if type(result) == dict:
                    cachedVersion = result.get("__cacheVersion")
                if cachedVersion != version:
                    log.info(f"Cached result has incorrect version ({cachedVersion}, expected {version}), recomputing ...")
                    result = callFnAndCacheResult()
                else:
                    result = result["obj"]
            return result
        else:
            log.info(f"Ignoring previously stored result in {picklePath}, calling function '{functionName}' ...")
            return callFnAndCacheResult()
    else:
        log.info(f"No cached result found in {picklePath}, calling function '{functionName}' ...")
        return callFnAndCacheResult()


class PickleCached(object):
    """
    Function decorator for caching function results via pickle
    """
    def __init__(self, cacheBasePath: str, filenamePrefix: str = None, filename: str = None, backend="pickle",
            protocol=pickle.HIGHEST_PROTOCOL, load=True, version=None):
        """
        :param cacheBasePath: the directory where the pickle cache file will be stored
        :param filenamePrefix: a prefix of the name of the cache file to be created, to which the function name and, where applicable,
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
        self.cacheBasePath = cacheBasePath
        self.filenamePrefix = filenamePrefix
        self.backend = backend
        self.protocol = protocol
        self.load = load
        self.version = version

        if self.filenamePrefix is None:
            self.filenamePrefix = ""
        else:
            self.filenamePrefix += "-"

    def __call__(self, fn: Callable, *_args, **_kwargs):

        def wrapped(*args, **kwargs):
            hashCodeStr = None
            haveArgs = len(args) > 0 or len(kwargs) > 0
            if haveArgs:
                hashCodeStr = pickleHash((args, kwargs))
            if self.filename is None:
                filename = self.filenamePrefix + fn.__qualname__.replace(".<locals>.", ".")
                if hashCodeStr is not None:
                    filename += "-" + hashCodeStr
                filename += ".cache.pickle"
            else:
                if hashCodeStr is not None:
                    if not "%s" in self.filename:
                        raise Exception("Function called with arguments but full cache filename contains no placeholder (%s) for argument hash")
                    filename = self.filename % hashCodeStr
                else:
                    if "%s" in self.filename:
                        raise Exception("Function without arguments but full cache filename with placeholder (%s) was specified")
                    filename = self.filename
            picklePath = os.path.join(self.cacheBasePath, filename)
            return cached(lambda: fn(*args, **kwargs), picklePath, functionName=fn.__name__, backend=self.backend, load=self.load,
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
    def save(self, path: str, backend="pickle"):
        """
        Saves the instance as pickle

        :param path:
        :param backend: pickle or joblib
        """
        dumpPickle(self, path, backend=backend)

    @classmethod
    def load(cls, path, backend="pickle"):
        """
        Loads a class instance from pickle

        :param path:
        :param backend: pickle or joblib
        :return: instance of the present class
        """
        log.info(f"Loading instance of {cls} from {path}")
        result = loadPickle(path, backend=backend)
        if not isinstance(result, cls):
            raise Exception(f"Excepted instance of {cls}, instead got: {result.__class__.__name__}")
        return result
