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

    def __init__(self, host, db, user, pw, value_type: ValueType, table_name="cache", deferred_commit_delay_secs=1.0, in_memory=False):
        import MySQLdb
        self.conn = MySQLdb.connect(host=host, database=db, user=user, password=pw)
        self.table_name = table_name
        self.max_key_length = 255
        self._update_hook = DelayedUpdateHook(self._commit, deferred_commit_delay_secs)
        self._num_entries_to_be_committed = 0

        cache_value_sql_type, self.is_cache_value_pickled = value_type.value

        cursor = self.conn.cursor()
        cursor.execute(f"SHOW TABLES;")
        if table_name not in [r[0] for r in cursor.fetchall()]:
            cursor.execute(f"CREATE TABLE {table_name} (cache_key VARCHAR({self.max_key_length}) PRIMARY KEY, "
                           f"cache_value {cache_value_sql_type});")
        cursor.close()

        self._in_memory_df = None if not in_memory else self._load_table_to_data_frame()

    def _load_table_to_data_frame(self):
        df = pd.read_sql(f"SELECT * FROM {self.table_name};", con=self.conn, index_col="cache_key")
        if self.is_cache_value_pickled:
            df["cache_value"] = df["cache_value"].apply(pickle.loads)
        return df

    def set(self, key, value):
        key = str(key)
        if len(key) > self.max_key_length:
            raise ValueError(f"Key too long, maximal key length is {self.max_key_length}")
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE cache_key=%s", (key,))
        stored_value = pickle.dumps(value) if self.is_cache_value_pickled else value
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"INSERT INTO {self.table_name} (cache_key, cache_value) VALUES (%s, %s)",
                (key, stored_value))
        else:
            cursor.execute(f"UPDATE {self.table_name} SET cache_value=%s WHERE cache_key=%s", (stored_value, key))
        self._num_entries_to_be_committed += 1
        self._update_hook.handle_update()
        cursor.close()
        if self._in_memory_df is not None:
            self._in_memory_df["cache_value"][str(key)] = value

    def get(self, key):
        value = self._get_from_in_memory_df(key)
        if value is None:
            value = self._get_from_table(key)
        return value

    def _get_from_table(self, key):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT cache_value FROM {self.table_name} WHERE cache_key=%s", (str(key),))
        row = cursor.fetchone()
        if row is None:
            return None
        stored_value = row[0]
        value = pickle.loads(stored_value) if self.is_cache_value_pickled else stored_value
        return value

    def _get_from_in_memory_df(self, key):
        if self._in_memory_df is None:
            return None
        try:
            return self._in_memory_df["cache_value"][str(key)]
        except Exception as e:
            log.debug(f"Unable to load value for key {str(key)} from in-memory dataframe: {e}")
            return None

    def _commit(self):
        log.info(f"Committing {self._num_entries_to_be_committed} cache entries to the database")
        self.conn.commit()
        self._num_entries_to_be_committed = 0
