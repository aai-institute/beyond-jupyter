import enum
import logging
import pickle
import pandas as pd

from .cache import PersistentKeyValueCache, DelayedUpdateHook

log = logging.getLogger(__name__)


class MySQLPersistentKeyValueCache(PersistentKeyValueCache):

    class ValueType(enum.Enum):
        DOUBLE = ("DOUBLE", False)  # (SQL data type, isCachedValuePickled)
        BLOB = ("BLOB", True)

    def __init__(self, host, db, user, pw, valueType: ValueType, tableName="cache", deferredCommitDelaySecs=1.0, inMemory=False):
        import MySQLdb
        self.conn = MySQLdb.connect(host=host, database=db, user=user, password=pw)
        self.tableName = tableName
        self.maxKeyLength = 255
        self._updateHook = DelayedUpdateHook(self._commit, deferredCommitDelaySecs)
        self._numEntriesToBeCommitted = 0

        cacheValueSqlType, self.isCacheValuePickled = valueType.value

        cursor = self.conn.cursor()
        cursor.execute(f"SHOW TABLES;")
        if tableName not in [r[0] for r in cursor.fetchall()]:
            cursor.execute(f"CREATE TABLE {tableName} (cache_key VARCHAR({self.maxKeyLength}) PRIMARY KEY, cache_value {cacheValueSqlType});")
        cursor.close()

        self._inMemoryDf = None if not inMemory else self._loadTableToDataFrame()

    def _loadTableToDataFrame(self):
        df = pd.read_sql(f"SELECT * FROM {self.tableName};", con=self.conn, index_col="cache_key")
        if self.isCacheValuePickled:
            df["cache_value"] = df["cache_value"].apply(pickle.loads)
        return df

    def set(self, key, value):
        key = str(key)
        if len(key) > self.maxKeyLength:
            raise ValueError(f"Key too long, maximal key length is {self.maxKeyLength}")
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.tableName} WHERE cache_key=%s", (key, ))
        storedValue = pickle.dumps(value) if self.isCacheValuePickled else value
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"INSERT INTO {self.tableName} (cache_key, cache_value) VALUES (%s, %s)",
                (key, storedValue))
        else:
            cursor.execute(f"UPDATE {self.tableName} SET cache_value=%s WHERE cache_key=%s", (storedValue, key))
        self._numEntriesToBeCommitted += 1
        self._updateHook.handleUpdate()
        cursor.close()
        if self._inMemoryDf is not None:
            self._inMemoryDf["cache_value"][str(key)] = value

    def get(self, key):
        value = self._getFromInMemoryDf(key)
        if value is None:
            value = self._getFromTable(key)
        return value

    def _getFromTable(self, key):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT cache_value FROM {self.tableName} WHERE cache_key=%s", (str(key), ))
        row = cursor.fetchone()
        if row is None:
            return None
        storedValue = row[0]
        value = pickle.loads(storedValue) if self.isCacheValuePickled else storedValue
        return value

    def _getFromInMemoryDf(self, key):
        if self._inMemoryDf is None:
            return None
        try:
            return self._inMemoryDf["cache_value"][str(key)]
        except Exception as e:
            log.debug(f"Unable to load value for key {str(key)} from in-memory dataframe: {e}")
            return None

    def _commit(self):
        log.info(f"Committing {self._numEntriesToBeCommitted} cache entries to the database")
        self.conn.commit()
        self._numEntriesToBeCommitted = 0
