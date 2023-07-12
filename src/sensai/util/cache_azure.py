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
    def deSerialise(self, value: str):
        pass


class NumpyArrayJsonSerialiser(Serialiser):
    """
    Serialises a numpy array as json string of list representation of array
    """

    def serialise(self, value: np.ndarray) -> str:
        return json.dumps(value.tolist())

    def deSerialise(self, value: str):
        return np.array(json.loads(value))


class PropertyLoader(ABC):
    """
    Abstraction of a customised loader for an entity property
    """

    @abstractmethod
    def loadPropertyValue(self, entity: Entity):
        pass

    @abstractmethod
    def writePropertyValue(self, entity: Entity):
        pass

    @abstractmethod
    def loadPropertyValueToDataFrameColumn(self, df: pd.DataFrame):
        pass


class SerialisedPropertyLoader(PropertyLoader):
    """
    PropertyLoader to serialise and de-serialise values. Useful, if type of values is not aligned with table storage data model,
    see https://docs.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model
    """
    def __init__(self, propertyName: str, serialiser: Serialiser):
        self.serialiser = serialiser
        self.propertyName = propertyName

    def loadPropertyValue(self, entity: Entity):
        entity[self.propertyName] = self.serialiser.deSerialise(entity[self.propertyName])

    def writePropertyValue(self, entity: Entity):
        entity[self.propertyName] = self.serialiser.serialise(entity[self.propertyName])

    def loadPropertyValueToDataFrameColumn(self, df: pd.DataFrame):
        if self.propertyName in df.columns:
            df.loc[:, self.propertyName] = [self.serialiser.deSerialise(value) for value in df[self.propertyName]]


class AzureTableBlobBackend(ABC):
    """
    Abstraction of a blob backend, which allows for convenient setting and getting of values stored in blob storage via a
    reference to the value
    """

    @abstractmethod
    def getValueFromReference(self, valueIdentifier: str):
        pass

    @abstractmethod
    def getValueReference(self, partitionKey: str, rowKey: str, valueName: str, blobNamePrefix: str = None) -> str:
        pass

    @abstractmethod
    def setValueForReference(self, valueIdentifier: str, value):
        pass


class BlobPerKeyAzureTableBlobBackend(AzureTableBlobBackend, ABC):

    """
    Backend stores serialised values as /tableName/partitionKey/rowKey/valueName.<fileExtension>
    or /tableName/rowKey/valueName.<fileExtension>, if partitionKey equals tableName
    """

    def __init__(self, blockBlobService: BlockBlobService, containerName: str):
        """

        :param blockBlobService: https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blockblobservice.blockblobservice?view=azure-python-previous
        """
        self.blockBlobService = blockBlobService
        self.containerName = containerName
        self.containerList = [container.name for container in blockBlobService.list_containers()]
        if containerName not in self.containerList:
            self.blockBlobService.create_container(containerName)
            self.containerList.append(containerName)

    @property
    @abstractmethod
    def fileExtension(self):
        pass

    @abstractmethod
    def _getBlobValue(self, containerName, blobName):
        pass

    @abstractmethod
    def _writeValueToBlob(self, containerName, blobName, value):
        pass

    def getValueFromReference(self, valueIdentifier: str):
        containerName = self._getContainerNameFromIdentifier(valueIdentifier)
        blobName = self._getBlobNameFromIdentifier(valueIdentifier)
        return self._getBlobValue(containerName, blobName)

    def getValueReference(self, partitionKey: str, rowKey: str, valueName: str, blobNamePrefix: str = None) -> str:
        blobName = self._getBlobNameFromKeys(partitionKey, rowKey, valueName, blobPrefix=blobNamePrefix)
        return self.blockBlobService.make_blob_url(self.containerName, blobName)

    def setValueForReference(self, valueIdentifier: str, value):
        containerName = self._getContainerNameFromIdentifier(valueIdentifier)
        blobName = self._getBlobNameFromIdentifier(valueIdentifier)
        self._writeValueToBlob(containerName, blobName, value)

    def _getBlobNameFromIdentifier(self, valueIdentifier: str):
        return (valueIdentifier.partition(f"{self.blockBlobService.primary_endpoint}/")[2]).partition("/")[2]

    def _getContainerNameFromIdentifier(self, valueIdentifier: str):
        return (valueIdentifier.partition(f"{self.blockBlobService.primary_endpoint}/")[2]).partition("/")[0]

    def _getBlobNameFromKeys(self, partitionKey: str, rowKey: str, valueName: str, blobPrefix: str = None):
        identifierList = [blobPrefix, partitionKey] if blobPrefix is not None and blobPrefix != partitionKey else [partitionKey]
        identifierList.extend([rowKey, valueName])
        return "/".join(identifierList) + self.fileExtension


class TextDumpAzureTableBlobBackend(BlobPerKeyAzureTableBlobBackend):
    """
   Backend stores values as txt files in the structure /tableName/partitionKey/rowKey/valueName
   """

    @property
    def fileExtension(self):
        return ""

    def _getBlobValue(self, containerName, blobName):
        return self.blockBlobService.get_blob_to_text(containerName, blobName).content

    def _writeValueToBlob(self, containerName, blobName, value):
        self.blockBlobService.create_blob_from_text(containerName, blobName, value)


class JsonAzureTableBlobBackend(BlobPerKeyAzureTableBlobBackend):
    """
    Backend stores values as json files in the structure /tableName/partitionKey/rowKey/valueName.json
    """

    @property
    def fileExtension(self):
        return ".json"

    def _getBlobValue(self, containerName, blobName):
        encodedValue = self.blockBlobService.get_blob_to_bytes(containerName, blobName).content
        return self._decodeBytesToValue(encodedValue)

    def _writeValueToBlob(self, containerName, blobName, value):
        encodedValue = self._encodeValueToBytes(value)
        self.blockBlobService.create_blob_from_bytes(containerName, blobName, encodedValue)

    @staticmethod
    def _encodeValueToBytes(value):
        return str.encode(json.dumps(value))

    @staticmethod
    def _decodeBytesToValue(_bytes):
        return json.loads(_bytes.decode())


class PickleAzureTableBlobBackend(JsonAzureTableBlobBackend):
    """
    Backend stores values as pickle files in the structure /tableName/partitionKey/rowKey/valueName.pickle
    """

    @property
    def fileExtension(self):
        return ".pickle"

    @staticmethod
    def _encodeValueToBytes(value):
        return pickle.dumps(value)

    @staticmethod
    def _decodeBytesToValue(_bytes):
        return pickle.loads(_bytes)


class BlobBackedPropertyLoader(PropertyLoader):
    AZURE_ALLOWED_SIZE_PER_PROPERTY_BYTES = 64000
    AZURE_ALLOWED_STRING_LENGTH_PER_PROPERTY = 32000

    """
    PropertyLoader to write and read values from blob backend via a reference to the value. Useful, if values cannot
    be stored in table storage itself, due to not being aligned with table storage data model, 
    see https://docs.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model
    """
    def __init__(self, propertyName: str, blobBackend: AzureTableBlobBackend, blobPrefix: str = None, propertyBooleanBlobStatusName: str = None, max_workers=None):
        """

        :param propertyName: name of property in table
        :param propertyBooleanBlobStatusName: name of property representing a boolean flag within a table, which indicates, if value is blob backed.
                                              If None, each value is assumed to be blob backed.
        :param blobBackend: actual backend to use for storage
        :param blobPrefix: prefix to use for blob in storage, e.g. a table name
        :param max_workers: maximal number of workers to load data from blob storage
        """
        self.blobPrefix = blobPrefix
        self.propertyBlobStatusName = propertyBooleanBlobStatusName
        self.blobBackend = blobBackend
        self.max_workers = max_workers
        self.propertyName = propertyName

    def loadPropertyValue(self, entity: Entity):
        if self._isEntityValueBlobBacked(entity):
            entity[self.propertyName] = self.blobBackend.getValueFromReference(entity[self.propertyName])

    def writePropertyValue(self, entity: Entity):
        if self.propertyName in entity.keys():
            if self._needToWriteToBlob(entity[self.propertyName]):
                valueIdentifier = self.blobBackend.getValueReference(entity["PartitionKey"], entity["RowKey"], self.propertyName, blobNamePrefix=self.blobPrefix)
                value = entity[self.propertyName]
                self.blobBackend.setValueForReference(valueIdentifier, value)
                entity[self.propertyName] = valueIdentifier
                propertyBlobStatus = True if self.propertyBlobStatusName is not None else None
            else:
                propertyBlobStatus = False if self.propertyBlobStatusName is not None else None

            if propertyBlobStatus is not None:
                entity[self.propertyBlobStatusName] = propertyBlobStatus

    def loadPropertyValueToDataFrameColumn(self, df: pd.DataFrame):
        if self.propertyName in df.columns:
            if self.propertyBlobStatusName is None:
                df.loc[:, self.propertyName] = self._loadValuesInSeries(df[self.propertyName])
            else:
                df.loc[df[self.propertyBlobStatusName], self.propertyName] = self._loadValuesInSeries(df.loc[df[self.propertyBlobStatusName], self.propertyName])

    def _needToWriteToBlob(self, value):
        if self.propertyBlobStatusName is None:
            return True
        if sys.getsizeof(value) > self.AZURE_ALLOWED_SIZE_PER_PROPERTY_BYTES:
            return True
        if isinstance(value, str) and len(value) > self.AZURE_ALLOWED_STRING_LENGTH_PER_PROPERTY:
            return True
        return False

    def _isEntityValueBlobBacked(self, entity: Entity):
        if self.propertyName not in entity.keys():
            return False
        if self.propertyBlobStatusName is None or self.propertyBlobStatusName not in entity:
            return True
        return entity[self.propertyBlobStatusName]

    def _loadValuesInSeries(self, _series: pd.Series):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            _series = list(executor.map(self.blobBackend.getValueFromReference, _series))
        return _series


class BlobBackedSerialisedPropertyLoader(BlobBackedPropertyLoader, SerialisedPropertyLoader):
    """
    Property loader, which combines serialisation and blob backing.
    """
    def __init__(self, propertyName, serialiser: Serialiser, blobBackend: AzureTableBlobBackend, blobPrefix: str = None,
            propertyBooleanBlobStatusName: str = None, max_workers=None):
        """


        :param propertyName: name of property in table
        :param serialiser:
        :param propertyBooleanBlobStatusName: name of property representing a boolean flag within a table, which indicates, if value is blob backed.
                                              If None, each value is assumed to be blob backed.
        :param blobBackend: actual backend to use for storage
        :param blobPrefix: prefix to use for blob in storage, e.g. a table name
        :param max_workers: maximal number of workers to load data from blob storage
        """
        SerialisedPropertyLoader.__init__(self, propertyName, serialiser)
        BlobBackedPropertyLoader.__init__(self, propertyName, blobBackend, blobPrefix, propertyBooleanBlobStatusName, max_workers)

    def loadPropertyValue(self, entity: Entity):
        super(BlobBackedPropertyLoader, self).loadPropertyValue(entity)
        super(SerialisedPropertyLoader, self).loadPropertyValue(entity)

    def writePropertyValue(self, entity: Entity):
        super(SerialisedPropertyLoader, self).writePropertyValue(entity)
        super(BlobBackedPropertyLoader, self).writePropertyValue(entity)

    def loadPropertyValueToDataFrameColumn(self, df: pd.DataFrame):
        super(BlobBackedPropertyLoader, self).loadPropertyValueToDataFrameColumn(df)
        super(SerialisedPropertyLoader, self).loadPropertyValueToDataFrameColumn(df)


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
            def __init__(self, partitionKey):
                self.partitionKey = partitionKey
                self._commandList = collections.deque()

            def __len__(self):
                return len(self._commandList)

            def append(self, command):
                self._commandList.append(command)

            def execute(self, contextManager: Callable[[], TableBatch],  batchSize: int):
                while len(self._commandList) > 0:
                    _slice = [self._commandList.popleft() for _ in range(min(batchSize, len(self._commandList)))]
                    _log.info(f"Committing {len(_slice)} cache entries to the database")
                    with contextManager() as batch:
                        for command in _slice:
                            command(batch)

        def __init__(self):
            self.partitionCommandsQueue = []
            self.partitionKey2Commands = {}
            self._threadLock = threading.Lock()

        def addCommand(self, partitionKey, command: Union[Callable[[TableBatch], Any], functools.partial[TableBatch]]):
            """
            Add a command to queue of corresponding partitionKey
            :param partitionKey:
            :param command: a callable on a TableBatch
            """
            with self._threadLock:
                if partitionKey not in self.partitionKey2Commands:
                    commands = self.PartitionCommands(partitionKey)
                    self.partitionCommandsQueue.append(commands)
                    self.partitionKey2Commands[partitionKey] = commands
                self.partitionKey2Commands[partitionKey].append(command)

        def pop(self, minLength: int = None) -> Optional[AzureLazyBatchCommitTable.PartitionCommandsPriorityQueue.PartitionCommands]:
            """
            :param minLength: minimal length of largest PartitionCommands for the pop to take place.
            :return: largest PartitionCommands or None if minimal length is not reached
            """
            with self._threadLock:
                return self._pop(minLength)

        def popAll(self):
            with self._threadLock:
                commandsList = []
                while not self._isEmpty():
                    commandsList.append(self._pop())
                return commandsList

        def isEmpty(self):
            with self._threadLock:
                return self._isEmpty()

        def _pop(self, minLength=None):
            length, index = self._getMaxPriorityInfo()
            if index is not None and (minLength is None or length >= minLength):
                q = self.partitionCommandsQueue.pop(index)
                del self.partitionKey2Commands[q.partitionKey]
                return q
            else:
                return None

        def _isEmpty(self):
            return len(self.partitionCommandsQueue) == 0

        def _getMaxPriorityInfo(self):
            lengthsList = list(map(len, self.partitionCommandsQueue))
            if len(lengthsList) == 0:
                return 0, None
            maxLength = max(lengthsList)
            return maxLength, lengthsList.index(maxLength)

    def __init__(self, tableName: str, tableService: TableService, propertyLoaders: Sequence[PropertyLoader] = ()):
        """

        :param tableName: name of table
        :param tableService: instance of :class:`azure.storage.table.TableService` to connect to Azure table storage
        :param propertyLoaders:
        """

        if not self.AZURE_ALLOWED_TABLE_NAME_PATTERN.match(tableName):
            raise ValueError(f"Invalid table name {tableName}, see: https://docs.microsoft.com/en-us/rest/api/storageservices/Understanding-the-Table-Service-Data-Model")

        self.tableService = tableService
        self.tableName = tableName
        self.propertyLoaders = propertyLoaders
        self._partitionQueues = self.PartitionCommandsPriorityQueue()
        self._contextManager = functools.partial(self.tableService.batch, self.tableName)

        if not self.exists():
            self.tableService.create_table(self.tableName)

    def insertOrReplaceEntity(self, entity: Union[Dict, Entity]):
        """
        Lazy wrapper method for :func:`azure.storage.table.TableService.insert_or_replace_entity`
        :param entity:
        """
        partitionKey = entity["PartitionKey"]
        for propertyLoader in self.propertyLoaders:
            propertyLoader.writePropertyValue(entity)
        executionCommand = functools.partial(self._insertOrReplaceEntityViaBatch, entity)
        self._partitionQueues.addCommand(partitionKey, executionCommand)

    def insertEntity(self, entity: Union[Dict, Entity]):
        """
        Lazy wrapper method for :func:`azure.storage.table.TableService.insert_entity`
        :param entity:
        """
        partitionKey = entity["PartitionKey"]
        for propertyLoader in self.propertyLoaders:
            propertyLoader.writePropertyValue(entity)
        executionCommand = functools.partial(self._insertEntityViaBatch, entity)
        self._partitionQueues.addCommand(partitionKey, executionCommand)

    def getEntity(self, partitionKey: str, rowKey: str) -> Optional[Entity]:
        """
        Wraps :func:`azure.storage.table.TableService.get_entity`
        :param partitionKey:
        :param rowKey:
        :return:
        """
        try:
            entity = self.tableService.get_entity(self.tableName, partitionKey, rowKey)
            for propertyLoader in self.propertyLoaders:
                propertyLoader.loadPropertyValue(entity)
            return entity
        except Exception as e:
            _log.debug(f"Unable to load value for partitionKey {partitionKey} and rowKey {rowKey} from table {self.tableName}: {e}")
            return None

    def commitBlockingUntilEmpty(self, maxBatchSize=AZURE_ALLOWED_TABLE_BATCH_SIZE):
        """
        Commit insertion commands. Commands are executed batch-wise per partition until partition queue is empty in a
        blocking manner.
        :param maxBatchSize: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        """

        maxBatchSize = self._validateMaxBatchSize(maxBatchSize)

        while not self._partitionQueues.isEmpty():
            commands = self._partitionQueues.pop()
            commands.execute(self._contextManager, maxBatchSize)

    def commitNonBlockingCurrentQueueState(self, maxBatchSize=AZURE_ALLOWED_TABLE_BATCH_SIZE):
        """
        Commit insertion commands. Empties the current PartitionCommandsQueue in a non blocking way.
        Commands are executed batch-wise per partition.
        :param maxBatchSize: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        """

        maxBatchSize = self._validateMaxBatchSize(maxBatchSize)

        def commit():
            commandsList = self._partitionQueues.popAll()
            for commands in commandsList:
                commands.execute(self._contextManager, maxBatchSize)

        thread = threading.Thread(target=commit, daemon=False)
        thread.start()

    def commitBlockingLargestPartitionFromQueue(self, maxBatchSize=AZURE_ALLOWED_TABLE_BATCH_SIZE, minLength=None):
        """
        Commits in a blocking way the largest partition from PartitionCommandsQueue
        :param maxBatchSize: maximal batch size to use for batch insertion, must be less or equal to batch size allowed by Azure
        :param minLength: minimal size of largest partition. If not None, pop and commit only if minLength is reached.
        :return:
        """
        maxBatchSize = self._validateMaxBatchSize(maxBatchSize)
        commands = self._partitionQueues.pop(minLength)
        if commands is not None:
            commands.execute(self._contextManager, maxBatchSize)

    def _validateMaxBatchSize(self, maxBatchSize):
        if maxBatchSize > self.AZURE_ALLOWED_TABLE_BATCH_SIZE:
            _log.warning(f"Provided maxBatchSize is larger than allowed size {self.AZURE_ALLOWED_TABLE_BATCH_SIZE}. Will use maxBatchSize {self.AZURE_ALLOWED_TABLE_BATCH_SIZE} instead.")
            maxBatchSize = self.AZURE_ALLOWED_TABLE_BATCH_SIZE
        return maxBatchSize

    def loadTableToDataFrame(self, columns: List[str] = None, rowFilterQuery: str = None, numRecords: int = None):
        """
        Load all rows of table to :class:`~pandas.DataFrame`
        :param rowFilterQuery:
        :param numRecords:
        :param columns: restrict loading to provided columns
        :return: :class:`~pandas.DataFrame`
        """
        if numRecords is None:
            records = list(self._iterRecords(columns, rowFilterQuery))
        else:
            records = []
            for record in self._iterRecords(columns, rowFilterQuery):
                records.append(record)
                if len(records) >= numRecords:
                    break
        df = pd.DataFrame(records, columns=columns)
        for propertyLoader in self.propertyLoaders:
            propertyLoader.loadPropertyValueToDataFrameColumn(df)
        return df

    def iterDataFrameChunks(self, chunkSize: int, columns: List[str] = None, rowFilterQuery: str = None):
        """
        Get a generator of dataframe chunks
        :param rowFilterQuery:
        :param chunkSize:
        :param columns:
        :return:
        """
        records = []
        for record in self._iterRecords(columns, rowFilterQuery):
            records.append(record)
            if len(records) >= chunkSize:
                df = pd.DataFrame(records, columns=columns)
                for propertyLoader in self.propertyLoaders:
                    propertyLoader.loadPropertyValueToDataFrameColumn(df)
                yield df
                records = []

    def iterRecords(self, columns: List[str] = None, rowFilterQuery: str = None):
        """

        Get a generator of table entities
        :param rowFilterQuery:
        :param columns:
        :return:
        """
        for entity in self._iterRecords(columns, rowFilterQuery):
            for propertyLoader in self.propertyLoaders:
                propertyLoader.loadPropertyValue(entity)
            yield entity

    def _iterRecords(self, columns: Optional[List[str]], rowFilterQuery: Optional[str]):

        columnNamesAsCommaSeparatedString = None
        if columns is not None:
            columnNamesAsCommaSeparatedString = ",".join(columns)
        return self.tableService.query_entities(self.tableName, select=columnNamesAsCommaSeparatedString,
                filter=rowFilterQuery)

    def insertDataFrameToTable(self, df: pd.DataFrame, partitionKeyGenerator: Callable[[str], str] = None, numRecords: int = None):
        """
        Inserts or replace entities of the table corresponding to rows of the DataFrame, where the index of the dataFrame acts as rowKey.
        Values of object type columns in the dataFrame may have to be serialised via json beforehand.
        :param df: DataFrame to be inserted
        :param partitionKeyGenerator: if None, partitionKeys default to tableName
        :param numRecords: restrict insertion to first numRecords rows, merely for testing
        """
        for (count, (idx, row)) in enumerate(df.iterrows()):
            if numRecords is not None:
                if count >= numRecords:
                    break
            entity = row.to_dict()
            entity["RowKey"] = idx
            entity["PartitionKey"] = self.tableName if partitionKeyGenerator is None else partitionKeyGenerator(idx)
            self.insertOrReplaceEntity(entity)

    @staticmethod
    def _insertOrReplaceEntityViaBatch(entity, batch: TableBatch):
        return batch.insert_or_replace_entity(entity)

    @staticmethod
    def _insertEntityViaBatch(entity, batch: TableBatch):
        return batch.insert_entity(entity)

    def exists(self):
        return self.tableService.exists(self.tableName)


class AzureTablePersistentKeyValueCache(PersistentKeyValueCache):
    """
    PersistentKeyValueCache using Azure Table Storage, see https://docs.microsoft.com/en-gb/azure/storage/tables/
    """
    CACHE_VALUE_IDENTIFIER = "cache_value"

    def __init__(self, tableService: TableService, tableName="cache", partitionKeyGenerator: Callable[[str], str] = None,
            maxBatchSize=100, minSizeForPeriodicCommit: Optional[int] = 100, deferredCommitDelaySecs=1.0, inMemory=False,
            blobBackend: AzureTableBlobBackend = None, serialiser: Serialiser = None, max_workers: int = None):
        """


        :param tableService: https://docs.microsoft.com/en-us/python/api/azure-cosmosdb-table/azure.cosmosdb.table.tableservice.tableservice?view=azure-python
        :param tableName: name of table, needs to match restrictions for Azure storage resources, see https://docs.microsoft.com/en-gb/azure/azure-resource-manager/management/resource-name-rules
        :param partitionKeyGenerator: callable to generate a partitionKey from provided string, if None partitionKey in requests defaults to tableName
        :param maxBatchSize: maximal batch size for each commit.
        :param deferredCommitDelaySecs: the time frame during which no new data must be added for a pending transaction to be committed
        :param minSizeForPeriodicCommit: minimal size of a batch to be committed in a periodic thread.
                                         If None, commits are only executed in a deferred manner, i.e. commit only if there is no update for deferredCommitDelaySecs
        :param inMemory: boolean flag, to indicate, if table should be loaded in memory at construction
        :param blobBackend: if not None, blob storage will be used to store actual value and cache_value in table only contains a reference
        :param max_workers: maximal number of workers to load data from blob backend
        """

        self._deferredCommitDelaySecs = deferredCommitDelaySecs
        self._partitionKeyGenerator = partitionKeyGenerator

        def createPropertyLoaders():
            if blobBackend is None and serialiser is None:
                _propertyLoaders = ()
            elif blobBackend is None and serialiser is not None:
                _propertyLoaders = (SerialisedPropertyLoader(self.CACHE_VALUE_IDENTIFIER, serialiser),)
            elif blobBackend is not None and serialiser is None:
                propertyBlobStatusName = self.CACHE_VALUE_IDENTIFIER + "_blob_backed"
                _propertyLoaders = (BlobBackedPropertyLoader(self.CACHE_VALUE_IDENTIFIER, blobBackend, tableName,
                    propertyBlobStatusName, max_workers),)
            else:
                propertyBlobStatusName = self.CACHE_VALUE_IDENTIFIER + "_blob_backed"
                _propertyLoaders = (BlobBackedSerialisedPropertyLoader(self.CACHE_VALUE_IDENTIFIER, serialiser, blobBackend,
                tableName, propertyBlobStatusName, max_workers),)
            return _propertyLoaders

        propertyLoaders = createPropertyLoaders()
        self._batchCommitTable = AzureLazyBatchCommitTable(tableName, tableService, propertyLoaders=propertyLoaders)
        self._minSizeForPeriodicCommit = minSizeForPeriodicCommit
        self._maxBatchSize = maxBatchSize
        self._updateHook = PeriodicUpdateHook(deferredCommitDelaySecs, noUpdateFn=self._commit, periodicFn=self._periodicallyCommit)

        self._inMemoryCache = None

        if inMemory:
            df = self._batchCommitTable.loadTableToDataFrame(columns=['RowKey', self.CACHE_VALUE_IDENTIFIER]).set_index("RowKey")
            _log.info(f"Loaded {len(df)} entries of table {tableName} in memory")
            self._inMemoryCache = df[self.CACHE_VALUE_IDENTIFIER].to_dict()

    def set(self, key, value):
        keyAsString = str(key)
        partitionKey = self._getPartitionKeyForRowKey(keyAsString)
        entity = {'PartitionKey': partitionKey, 'RowKey': keyAsString, self.CACHE_VALUE_IDENTIFIER: value}
        self._batchCommitTable.insertOrReplaceEntity(entity)
        self._updateHook.handleUpdate()

        if self._inMemoryCache is not None:
            self._inMemoryCache[keyAsString] = value

    def get(self, key):
        keyAsString = str(key)
        value = self._getFromInMemoryCache(keyAsString)
        if value is None:
            value = self._getFromTable(keyAsString)
        return value

    def _getFromTable(self, key: str):
        partitionKey = self._getPartitionKeyForRowKey(key)
        entity = self._batchCommitTable.getEntity(partitionKey, key)
        if entity is not None:
            return entity[self.CACHE_VALUE_IDENTIFIER]
        return None

    def _getFromInMemoryCache(self, key):
        if self._inMemoryCache is None:
            return None
        return self._inMemoryCache.get(str(key), None)

    def _getPartitionKeyForRowKey(self, key: str):
        return self._batchCommitTable.tableName if self._partitionKeyGenerator is None else self._partitionKeyGenerator(key)

    def _commit(self):
        self._batchCommitTable.commitNonBlockingCurrentQueueState(self._maxBatchSize)

    def _periodicallyCommit(self):
        self._batchCommitTable.commitBlockingLargestPartitionFromQueue(self._maxBatchSize, self._minSizeForPeriodicCommit)
