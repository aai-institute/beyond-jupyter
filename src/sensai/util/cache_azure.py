from __future__ import annotations
import collections
import functools
import pickle
import sys
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable, Dict, Union, Any, List, Sequence, Optional
import json
import logging
import re
import threading


from azure.storage.table import TableService, TableBatch, Entity
from azure.storage.blob import BlockBlobService
import pandas as pd
import numpy as np

from .cache import PersistentKeyValueCache, PeriodicUpdateHook

_log = logging.getLogger(__name__)


class Serialiser(ABC):
    """
    Abstraction for mechanisms to serialise values, which do not fit table storage data model,
    see https://docs.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model
    """

    @abstractmethod
    def serialise(self, value) -> str:
        pass

    @abstractmethod
    def deserialise(self, value: str):
        pass


class NumpyArrayJsonSerialiser(Serialiser):
    """
    Serialises a numpy array as json string of list representation of array
    """

    def serialise(self, value: np.ndarray) -> str:
        return json.dumps(value.tolist())

    def deserialise(self, value: str):
        return np.array(json.loads(value))


class PropertyLoader(ABC):
    """
    Abstraction of a customised loader for an entity property
    """

    @abstractmethod
    def load_property_value(self, entity: Entity):
        pass

    @abstractmethod
    def write_property_value(self, entity: Entity):
        pass

    @abstractmethod
    def load_property_value_to_data_frame_column(self, df: pd.DataFrame):
        pass


class SerialisedPropertyLoader(PropertyLoader):
    """
    PropertyLoader to serialise and de-serialise values. Useful, if type of values is not aligned with table storage data model,
    see https://docs.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model
    """
    def __init__(self, property_name: str, serialiser: Serialiser):
        self.serialiser = serialiser
        self.property_name = property_name

    def load_property_value(self, entity: Entity):
        entity[self.property_name] = self.serialiser.deserialise(entity[self.property_name])

    def write_property_value(self, entity: Entity):
        entity[self.property_name] = self.serialiser.serialise(entity[self.property_name])

    def load_property_value_to_data_frame_column(self, df: pd.DataFrame):
        if self.property_name in df.columns:
            df.loc[:, self.property_name] = [self.serialiser.deserialise(value) for value in df[self.property_name]]


class AzureTableBlobBackend(ABC):
    """
    Abstraction of a blob backend, which allows for convenient setting and getting of values stored in blob storage via a
    reference to the value
    """

    @abstractmethod
    def get_value_from_reference(self, value_identifier: str):
        pass

    @abstractmethod
    def get_value_reference(self, partition_key: str, row_key: str, value_name: str, blob_name_prefix: str = None) -> str:
        pass

    @abstractmethod
    def set_value_for_reference(self, value_identifier: str, value):
        pass


class BlobPerKeyAzureTableBlobBackend(AzureTableBlobBackend, ABC):

    """
    Backend stores serialised values as /tableName/partitionKey/rowKey/valueName.<fileExtension>
    or /tableName/rowKey/valueName.<fileExtension>, if partitionKey equals tableName
    """

    def __init__(self, block_blob_service: BlockBlobService, container_name: str):
        """

        :param block_blob_service: https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blockblobservice.blockblobservice?view=azure-python-previous
        """
        self.block_blob_service = block_blob_service
        self.container_name = container_name
        self.container_list = [container.name for container in block_blob_service.list_containers()]
        if container_name not in self.container_list:
            self.block_blob_service.create_container(container_name)
            self.container_list.append(container_name)

    @property
    @abstractmethod
    def file_extension(self):
        pass

    @abstractmethod
    def _get_blob_value(self, container_name, blob_name):
        pass

    @abstractmethod
    def _write_value_to_blob(self, container_name, blob_name, value):
        pass

    def get_value_from_reference(self, value_identifier: str):
        container_name = self._get_container_name_from_identifier(value_identifier)
        blob_name = self._get_blob_name_from_identifier(value_identifier)
        return self._get_blob_value(container_name, blob_name)

    def get_value_reference(self, partition_key: str, row_key: str, value_name: str, blob_name_prefix: str = None) -> str:
        blob_name = self._get_blob_name_from_keys(partition_key, row_key, value_name, blob_prefix=blob_name_prefix)
        return self.block_blob_service.make_blob_url(self.container_name, blob_name)

    def set_value_for_reference(self, value_identifier: str, value):
        container_name = self._get_container_name_from_identifier(value_identifier)
        blob_name = self._get_blob_name_from_identifier(value_identifier)
        self._write_value_to_blob(container_name, blob_name, value)

    def _get_blob_name_from_identifier(self, value_identifier: str):
        return (value_identifier.partition(f"{self.block_blob_service.primary_endpoint}/")[2]).partition("/")[2]

    def _get_container_name_from_identifier(self, value_identifier: str):
        return (value_identifier.partition(f"{self.block_blob_service.primary_endpoint}/")[2]).partition("/")[0]

    def _get_blob_name_from_keys(self, partition_key: str, row_key: str, value_name: str, blob_prefix: str = None):
        identifier_list = [blob_prefix, partition_key] if blob_prefix is not None and blob_prefix != partition_key else [partition_key]
        identifier_list.extend([row_key, value_name])
        return "/".join(identifier_list) + self.file_extension


class TextDumpAzureTableBlobBackend(BlobPerKeyAzureTableBlobBackend):
    """
   Backend stores values as txt files in the structure /tableName/partitionKey/rowKey/valueName
   """

    @property
    def file_extension(self):
        return ""

    def _get_blob_value(self, container_name, blob_name):
        return self.block_blob_service.get_blob_to_text(container_name, blob_name).content

    def _write_value_to_blob(self, container_name, blob_name, value):
        self.block_blob_service.create_blob_from_text(container_name, blob_name, value)


class JsonAzureTableBlobBackend(BlobPerKeyAzureTableBlobBackend):
    """
    Backend stores values as json files in the structure /tableName/partitionKey/rowKey/valueName.json
    """

    @property
    def file_extension(self):
        return ".json"

    def _get_blob_value(self, container_name, blob_name):
        encoded_value = self.block_blob_service.get_blob_to_bytes(container_name, blob_name).content
        return self._decode_bytes_to_value(encoded_value)

    def _write_value_to_blob(self, container_name, blob_name, value):
        encoded_value = self._encode_value_to_bytes(value)
        self.block_blob_service.create_blob_from_bytes(container_name, blob_name, encoded_value)

    @staticmethod
    def _encode_value_to_bytes(value):
        return str.encode(json.dumps(value))

    @staticmethod
    def _decode_bytes_to_value(_bytes):
        return json.loads(_bytes.decode())


class PickleAzureTableBlobBackend(JsonAzureTableBlobBackend):
    """
    Backend stores values as pickle files in the structure /tableName/partitionKey/rowKey/valueName.pickle
    """

    @property
    def file_extension(self):
        return ".pickle"

    @staticmethod
    def _encode_value_to_bytes(value):
        return pickle.dumps(value)

    @staticmethod
    def _decode_bytes_to_value(_bytes):
        return pickle.loads(_bytes)


class BlobBackedPropertyLoader(PropertyLoader):
    AZURE_ALLOWED_SIZE_PER_PROPERTY_BYTES = 64000
    AZURE_ALLOWED_STRING_LENGTH_PER_PROPERTY = 32000

    """
    PropertyLoader to write and read values from blob backend via a reference to the value. Useful, if values cannot
    be stored in table storage itself, due to not being aligned with table storage data model, 
    see https://docs.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model
    """
    def __init__(self, property_name: str, blob_backend: AzureTableBlobBackend, blob_prefix: str = None,
            property_boolean_blob_status_name: str = None, max_workers=None):
        """
        :param property_name: name of property in table
        :param property_boolean_blob_status_name: name of property representing a boolean flag within a table, which indicates, if value is
            blob backed. If None, each value is assumed to be blob backed.
        :param blob_backend: actual backend to use for storage
        :param blob_prefix: prefix to use for blob in storage, e.g. a table name
        :param max_workers: maximal number of workers to load data from blob storage
        """
        self.blob_prefix = blob_prefix
        self.property_blob_status_name = property_boolean_blob_status_name
        self.blob_backend = blob_backend
        self.max_workers = max_workers
        self.propertyName = property_name

    def load_property_value(self, entity: Entity):
        if self._is_entity_value_blob_backed(entity):
            entity[self.propertyName] = self.blob_backend.get_value_from_reference(entity[self.propertyName])

    def write_property_value(self, entity: Entity):
        if self.propertyName in entity.keys():
            if self._need_to_write_to_blob(entity[self.propertyName]):
                value_identifier = self.blob_backend.get_value_reference(entity["PartitionKey"], entity["RowKey"], self.propertyName,
                    blob_name_prefix=self.blob_prefix)
                value = entity[self.propertyName]
                self.blob_backend.set_value_for_reference(value_identifier, value)
                entity[self.propertyName] = value_identifier
                property_blob_status = True if self.property_blob_status_name is not None else None
            else:
                property_blob_status = False if self.property_blob_status_name is not None else None

            if property_blob_status is not None:
                entity[self.property_blob_status_name] = property_blob_status

    def load_property_value_to_data_frame_column(self, df: pd.DataFrame):
        if self.propertyName in df.columns:
            if self.property_blob_status_name is None:
                df.loc[:, self.propertyName] = self._load_values_in_series(df[self.propertyName])
            else:
                df.loc[df[self.property_blob_status_name], self.propertyName] = \
                    self._load_values_in_series(df.loc[df[self.property_blob_status_name], self.propertyName])

    def _need_to_write_to_blob(self, value):
        if self.property_blob_status_name is None:
            return True
        if sys.getsizeof(value) > self.AZURE_ALLOWED_SIZE_PER_PROPERTY_BYTES:
            return True
        if isinstance(value, str) and len(value) > self.AZURE_ALLOWED_STRING_LENGTH_PER_PROPERTY:
            return True
        return False

    def _is_entity_value_blob_backed(self, entity: Entity):
        if self.propertyName not in entity.keys():
            return False
        if self.property_blob_status_name is None or self.property_blob_status_name not in entity:
            return True
        return entity[self.property_blob_status_name]

    def _load_values_in_series(self, _series: pd.Series):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            _series = list(executor.map(self.blob_backend.get_value_from_reference, _series))
        return _series


class BlobBackedSerialisedPropertyLoader(BlobBackedPropertyLoader, SerialisedPropertyLoader):
    """
    Property loader, which combines serialisation and blob backing.
    """
    def __init__(self, property_name, serialiser: Serialiser, blob_backend: AzureTableBlobBackend, blob_prefix: str = None,
            property_boolean_blob_status_name: str = None, max_workers=None):
        """


        :param property_name: name of property in table
        :param serialiser:
        :param property_boolean_blob_status_name: name of property representing a boolean flag within a table, which indicates, if value is
            blob backed. If None, each value is assumed to be blob backed.
        :param blob_backend: actual backend to use for storage
        :param blob_prefix: prefix to use for blob in storage, e.g. a table name
        :param max_workers: maximal number of workers to load data from blob storage
        """
        SerialisedPropertyLoader.__init__(self, property_name, serialiser)
        BlobBackedPropertyLoader.__init__(self, property_name, blob_backend, blob_prefix, property_boolean_blob_status_name, max_workers)

    def load_property_value(self, entity: Entity):
        super(BlobBackedPropertyLoader, self).load_property_value(entity)
        super(SerialisedPropertyLoader, self).load_property_value(entity)

    def write_property_value(self, entity: Entity):
        super(SerialisedPropertyLoader, self).write_property_value(entity)
        super(BlobBackedPropertyLoader, self).write_property_value(entity)

    def load_property_value_to_data_frame_column(self, df: pd.DataFrame):
        super(BlobBackedPropertyLoader, self).load_property_value_to_data_frame_column(df)
        super(SerialisedPropertyLoader, self).load_property_value_to_data_frame_column(df)


class AzureLazyBatchCommitTable:
    """
    Wrapper for an Azure table, which allow for convenient insertion via lazy batch execution per partition.
    Uses a priority queue to manage order of partitions to be committed.
    To execute insertions, call :func:`LazyBatchCommitTable.commit`
    """

    AZURE_ALLOWED_TABLE_NAME_PATTERN = re.compile("^[A-Za-z][A-Za-z0-9]{2,62}$")
    AZURE_ALLOWED_TABLE_BATCH_SIZE = 100

    class PartitionCommandsPriorityQueue:

        class PartitionCommands:
            def __init__(self, partition_key):
                self.partition_key = partition_key
                self._command_list = collections.deque()

            def __len__(self):
                return len(self._command_list)

            def append(self, command):
                self._command_list.append(command)

            def execute(self, context_manager: Callable[[], TableBatch], batch_size: int):
                while len(self._command_list) > 0:
                    _slice = [self._command_list.popleft() for _ in range(min(batch_size, len(self._command_list)))]
                    _log.info(f"Committing {len(_slice)} cache entries to the database")
                    with context_manager() as batch:
                        for command in _slice:
                            command(batch)

        def __init__(self):
            self.partition_commands_queue = []
            self.partition_key2_commands = {}
            self._thread_lock = threading.Lock()

        def add_command(self, partition_key, command: Union[Callable[[TableBatch], Any], functools.partial[TableBatch]]):
            """
            Add a command to queue of corresponding partitionKey
            :param partition_key:
            :param command: a callable on a TableBatch
            """
            with self._thread_lock:
                if partition_key not in self.partition_key2_commands:
                    commands = self.PartitionCommands(partition_key)
                    self.partition_commands_queue.append(commands)
                    self.partition_key2_commands[partition_key] = commands
                self.partition_key2_commands[partition_key].append(command)

        def pop(self, min_length: int = None) -> Optional[AzureLazyBatchCommitTable.PartitionCommandsPriorityQueue.PartitionCommands]:
            """
            :param min_length: minimal length of largest PartitionCommands for the pop to take place.
            :return: largest PartitionCommands or None if minimal length is not reached
            """
            with self._thread_lock:
                return self._pop(min_length)

        def pop_all(self):
            with self._thread_lock:
                commands_list = []
                while not self._is_empty():
                    commands_list.append(self._pop())
                return commands_list

        def is_empty(self):
            with self._thread_lock:
                return self._is_empty()

        def _pop(self, min_length=None):
            length, index = self._get_max_priority_info()
            if index is not None and (min_length is None or length >= min_length):
                q = self.partition_commands_queue.pop(index)
                del self.partition_key2_commands[q.partition_key]
                return q
            else:
                return None

        def _is_empty(self):
            return len(self.partition_commands_queue) == 0

        def _get_max_priority_info(self):
            lengths_list = list(map(len, self.partition_commands_queue))
            if len(lengths_list) == 0:
                return 0, None
            max_length = max(lengths_list)
            return max_length, lengths_list.index(max_length)

    def __init__(self, table_name: str, table_service: TableService, property_loaders: Sequence[PropertyLoader] = ()):
        """
        :param table_name: name of table
        :param table_service: instance of :class:`azure.storage.table.TableService` to connect to Azure table storage
        :param property_loaders:
        """

        if not self.AZURE_ALLOWED_TABLE_NAME_PATTERN.match(table_name):
            raise ValueError(f"Invalid table name {table_name}, see: "
                             f"https://docs.microsoft.com/en-us/rest/api/storageservices/Understanding-the-Table-Service-Data-Model")

        self.table_service = table_service
        self.table_name = table_name
        self.property_loaders = property_loaders
        self._partition_queues = self.PartitionCommandsPriorityQueue()
        self._context_manager = functools.partial(self.table_service.batch, self.table_name)

        if not self.exists():
            self.table_service.create_table(self.table_name)

    def insert_or_replace_entity(self, entity: Union[Dict, Entity]):
        """
        Lazy wrapper method for :func:`azure.storage.table.TableService.insert_or_replace_entity`
        :param entity:
        """
        partition_key = entity["PartitionKey"]
        for property_loader in self.property_loaders:
            property_loader.write_property_value(entity)
        execution_command = functools.partial(self._insert_or_replace_entity_via_batch, entity)
        self._partition_queues.add_command(partition_key, execution_command)

    def insert_entity(self, entity: Union[Dict, Entity]):
        """
        Lazy wrapper method for :func:`azure.storage.table.TableService.insert_entity`
        :param entity:
        """
        partition_key = entity["PartitionKey"]
        for property_loader in self.property_loaders:
            property_loader.write_property_value(entity)
        execution_command = functools.partial(self._insert_entity_via_batch, entity)
        self._partition_queues.add_command(partition_key, execution_command)

    def get_entity(self, partition_key: str, row_key: str) -> Optional[Entity]:
        """
        Wraps :func:`azure.storage.table.TableService.get_entity`
        :param partition_key:
        :param row_key:
        :return:
        """
        try:
            entity = self.table_service.get_entity(self.table_name, partition_key, row_key)
            for property_loader in self.property_loaders:
                property_loader.load_property_value(entity)
            return entity
        except Exception as e:
            _log.debug(f"Unable to load value for partitionKey {partition_key} and rowKey {row_key} from table {self.table_name}: {e}")
            return None

    def commit_blocking_until_empty(self, max_batch_size=AZURE_ALLOWED_TABLE_BATCH_SIZE):
        """
        Commit insertion commands. Commands are executed batch-wise per partition until partition queue is empty in a
        blocking manner.
        :param max_batch_size: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        """

        max_batch_size = self._validate_max_batch_size(max_batch_size)

        while not self._partition_queues.is_empty():
            commands = self._partition_queues.pop()
            commands.execute(self._context_manager, max_batch_size)

    def commit_non_blocking_current_queue_state(self, max_batch_size=AZURE_ALLOWED_TABLE_BATCH_SIZE):
        """
        Commit insertion commands. Empties the current PartitionCommandsQueue in a non blocking way.
        Commands are executed batch-wise per partition.
        :param max_batch_size: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        """

        max_batch_size = self._validate_max_batch_size(max_batch_size)

        def commit():
            commands_list = self._partition_queues.pop_all()
            for commands in commands_list:
                commands.execute(self._context_manager, max_batch_size)

        thread = threading.Thread(target=commit, daemon=False)
        thread.start()

    def commit_blocking_largest_partition_from_queue(self, max_batch_size=AZURE_ALLOWED_TABLE_BATCH_SIZE, min_length=None):
        """
        Commits in a blocking way the largest partition from PartitionCommandsQueue
        :param max_batch_size: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        :param min_length: minimal size of largest partition. If not None, pop and commit only if minLength is reached.
        :return:
        """
        max_batch_size = self._validate_max_batch_size(max_batch_size)
        commands = self._partition_queues.pop(min_length)
        if commands is not None:
            commands.execute(self._context_manager, max_batch_size)

    def _validate_max_batch_size(self, max_batch_size):
        if max_batch_size > self.AZURE_ALLOWED_TABLE_BATCH_SIZE:
            _log.warning(f"Provided maxBatchSize is larger than allowed size {self.AZURE_ALLOWED_TABLE_BATCH_SIZE}. "
                         f"Will use maxBatchSize {self.AZURE_ALLOWED_TABLE_BATCH_SIZE} instead.")
            max_batch_size = self.AZURE_ALLOWED_TABLE_BATCH_SIZE
        return max_batch_size

    def load_table_to_data_frame(self, columns: List[str] = None, row_filter_query: str = None, num_records: int = None):
        """
        Load all rows of table to :class:`~pandas.DataFrame`
        :param row_filter_query:
        :param num_records:
        :param columns: restrict loading to provided columns
        :return: :class:`~pandas.DataFrame`
        """
        if num_records is None:
            records = list(self._iter_records(columns, row_filter_query))
        else:
            records = []
            for record in self._iter_records(columns, row_filter_query):
                records.append(record)
                if len(records) >= num_records:
                    break
        df = pd.DataFrame(records, columns=columns)
        for property_loader in self.property_loaders:
            property_loader.load_property_value_to_data_frame_column(df)
        return df

    def iter_data_frame_chunks(self, chunk_size: int, columns: List[str] = None, row_filter_query: str = None):
        """
        Get a generator of dataframe chunks
        :param row_filter_query:
        :param chunk_size:
        :param columns:
        :return:
        """
        records = []
        for record in self._iter_records(columns, row_filter_query):
            records.append(record)
            if len(records) >= chunk_size:
                df = pd.DataFrame(records, columns=columns)
                for propertyLoader in self.property_loaders:
                    propertyLoader.load_property_value_to_data_frame_column(df)
                yield df
                records = []

    def iter_records(self, columns: List[str] = None, row_filter_query: str = None):
        """

        Get a generator of table entities
        :param row_filter_query:
        :param columns:
        :return:
        """
        for entity in self._iter_records(columns, row_filter_query):
            for propertyLoader in self.property_loaders:
                propertyLoader.load_property_value(entity)
            yield entity

    def _iter_records(self, columns: Optional[List[str]], row_filter_query: Optional[str]):
        column_names_as_comma_separated_string = None
        if columns is not None:
            column_names_as_comma_separated_string = ",".join(columns)
        return self.table_service.query_entities(self.table_name, select=column_names_as_comma_separated_string,
                filter=row_filter_query)

    def insert_data_frame_to_table(self, df: pd.DataFrame, partition_key_generator: Callable[[str], str] = None, num_records: int = None):
        """
        Inserts or replace entities of the table corresponding to rows of the DataFrame, where the index of the dataFrame acts as rowKey.
        Values of object type columns in the dataFrame may have to be serialised via json beforehand.
        :param df: DataFrame to be inserted
        :param partition_key_generator: if None, partitionKeys default to tableName
        :param num_records: restrict insertion to first numRecords rows, merely for testing
        """
        for (count, (idx, row)) in enumerate(df.iterrows()):
            if num_records is not None:
                if count >= num_records:
                    break
            entity = row.to_dict()
            entity["RowKey"] = idx
            entity["PartitionKey"] = self.table_name if partition_key_generator is None else partition_key_generator(idx)
            self.insert_or_replace_entity(entity)

    @staticmethod
    def _insert_or_replace_entity_via_batch(entity, batch: TableBatch):
        return batch.insert_or_replace_entity(entity)

    @staticmethod
    def _insert_entity_via_batch(entity, batch: TableBatch):
        return batch.insert_entity(entity)

    def exists(self):
        return self.table_service.exists(self.table_name)


class AzureTablePersistentKeyValueCache(PersistentKeyValueCache):
    """
    PersistentKeyValueCache using Azure Table Storage, see https://docs.microsoft.com/en-gb/azure/storage/tables/
    """
    CACHE_VALUE_IDENTIFIER = "cache_value"

    def __init__(self, table_service: TableService, table_name="cache", partition_key_generator: Callable[[str], str] = None,
            max_batch_size=100, min_size_for_periodic_commit: Optional[int] = 100, deferred_commit_delay_secs=1.0, in_memory=False,
            blob_backend: AzureTableBlobBackend = None, serialiser: Serialiser = None, max_workers: int = None):
        """
        :param table_service: https://docs.microsoft.com/en-us/python/api/azure-cosmosdb-table/azure.cosmosdb.table.tableservice.tableservice?view=azure-python
        :param table_name: name of table, needs to match restrictions for Azure storage resources, see https://docs.microsoft.com/en-gb/azure/azure-resource-manager/management/resource-name-rules
        :param partition_key_generator: callable to generate a partitionKey from provided string, if None partitionKey in requests defaults
            to tableName
        :param max_batch_size: maximal batch size for each commit.
        :param deferred_commit_delay_secs: the time frame during which no new data must be added for a pending transaction to be committed
        :param min_size_for_periodic_commit: minimal size of a batch to be committed in a periodic thread.
            If None, commits are only executed in a deferred manner, i.e. commit only if there is no update for `deferred_commit_delay_secs`
        :param in_memory: boolean flag, to indicate, if table should be loaded in memory at construction
        :param blob_backend: if not None, blob storage will be used to store actual value and cache_value in table only contains a reference
        :param max_workers: maximal number of workers to load data from blob backend
        """

        self._deferredCommitDelaySecs = deferred_commit_delay_secs
        self._partitionKeyGenerator = partition_key_generator

        def create_property_loaders():
            if blob_backend is None and serialiser is None:
                _property_loaders = ()
            elif blob_backend is None and serialiser is not None:
                _property_loaders = (SerialisedPropertyLoader(self.CACHE_VALUE_IDENTIFIER, serialiser),)
            elif blob_backend is not None and serialiser is None:
                property_blob_status_name = self.CACHE_VALUE_IDENTIFIER + "_blob_backed"
                _property_loaders = (BlobBackedPropertyLoader(self.CACHE_VALUE_IDENTIFIER, blob_backend, table_name,
                    property_blob_status_name, max_workers),)
            else:
                property_blob_status_name = self.CACHE_VALUE_IDENTIFIER + "_blob_backed"
                _property_loaders = (BlobBackedSerialisedPropertyLoader(self.CACHE_VALUE_IDENTIFIER, serialiser, blob_backend,
                table_name, property_blob_status_name, max_workers),)
            return _property_loaders

        property_loaders = create_property_loaders()
        self._batch_commit_table = AzureLazyBatchCommitTable(table_name, table_service, property_loaders=property_loaders)
        self._minSizeForPeriodicCommit = min_size_for_periodic_commit
        self._maxBatchSize = max_batch_size
        self._updateHook = PeriodicUpdateHook(deferred_commit_delay_secs, no_update_fn=self._commit, periodic_fn=self._periodically_commit)

        self._in_memory_cache = None

        if in_memory:
            df = self._batch_commit_table.load_table_to_data_frame(columns=['RowKey', self.CACHE_VALUE_IDENTIFIER]).set_index("RowKey")
            _log.info(f"Loaded {len(df)} entries of table {table_name} in memory")
            self._in_memory_cache = df[self.CACHE_VALUE_IDENTIFIER].to_dict()

    def set(self, key, value):
        key_as_string = str(key)
        partition_key = self._get_partition_key_for_row_key(key_as_string)
        entity = {'PartitionKey': partition_key, 'RowKey': key_as_string, self.CACHE_VALUE_IDENTIFIER: value}
        self._batch_commit_table.insert_or_replace_entity(entity)
        self._updateHook.handle_update()

        if self._in_memory_cache is not None:
            self._in_memory_cache[key_as_string] = value

    def get(self, key):
        key_as_string = str(key)
        value = self._get_from_in_memory_cache(key_as_string)
        if value is None:
            value = self._get_from_table(key_as_string)
        return value

    def _get_from_table(self, key: str):
        partition_key = self._get_partition_key_for_row_key(key)
        entity = self._batch_commit_table.get_entity(partition_key, key)
        if entity is not None:
            return entity[self.CACHE_VALUE_IDENTIFIER]
        return None

    def _get_from_in_memory_cache(self, key):
        if self._in_memory_cache is None:
            return None
        return self._in_memory_cache.get(str(key), None)

    def _get_partition_key_for_row_key(self, key: str):
        return self._batch_commit_table.table_name if self._partitionKeyGenerator is None else self._partitionKeyGenerator(key)

    def _commit(self):
        self._batch_commit_table.commit_non_blocking_current_queue_state(self._maxBatchSize)

    def _periodically_commit(self):
        self._batch_commit_table.commit_blocking_largest_partition_from_queue(self._maxBatchSize, self._minSizeForPeriodicCommit)
