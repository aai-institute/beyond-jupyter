from typing import Dict, Generator, Tuple, Optional, Union

import pandas as pd
import torch
import torchtext

from .torch_data import toTensor, TorchDataSet, TorchDataSetProvider


class TorchtextDataSetFromDataFrame(torchtext.data.Dataset):
    """
    A specialisation of torchtext.data.Dataset, where the data is taken from a pandas.DataFrame
    """
    def __init__(self, df: pd.DataFrame, fields: Dict[str, torchtext.data.Field]):
        """
        :param df: the data frame from which to obtain the data
        :param fields: a mapping from column names in the given data frame to torchtext fields, i.e.
            the keys are the columns to read and the values are the fields to use for generated Example instances
        """
        examples = df.apply(self._exampleFromSeries, args=(fields,), axis=1).tolist()
        fields = dict(fields)
        super().__init__(examples, fields)

    @classmethod
    def _exampleFromSeries(cls, series: pd.Series, fields: Dict[str, torchtext.data.Field]):
        return cls._exampleFromDict(series.to_dict(), fields)

    @classmethod
    def _exampleFromDict(cls, d: dict, fields: Dict[str, torchtext.data.Field]):
        ex = torchtext.data.Example()
        for key, field in fields.items():
            if key not in d:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(d[key]))
            else:
                setattr(ex, key, d[key])
        return ex


class TorchDataSetFromTorchtextDataSet(TorchDataSet):
    def __init__(self, dataSet: torchtext.data.Dataset, inputField: str, outputField: Optional[str], cuda: bool):
        self.outputField = outputField
        self.inputField = inputField
        self.dataSet = dataSet
        self.cuda = cuda

    def iterBatches(self, batchSize: int, shuffle: bool = False, inputOnly=False) -> Generator[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None, None]:
        iterator = torchtext.data.BucketIterator(self.dataSet,
            batch_size=batchSize,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False)

        for batch in iterator:
            x = toTensor(getattr(batch, self.inputField), self.cuda)
            if not inputOnly and self.outputField is not None:
                y = toTensor(getattr(batch, self.outputField), self.cuda)
                yield x, y
            else:
                yield x

    def size(self) -> Optional[int]:
        return len(self.dataSet)


class TorchDataSetProviderFromTorchtextDataSet(TorchDataSetProvider):
    def __init__(self, dataSet: torchtext.data.Dataset, inputField: str, outputField: str, cuda: bool, modelOutputDim, inputDim=None):
        super().__init__(modelOutputDim=modelOutputDim, inputDim=inputDim)
        self.dataSet = dataSet
        self.outputField = outputField
        self.inputField = inputField
        self.cuda = cuda

    def provideSplit(self, fractionalSizeOfFirstSet: float) -> Tuple[TorchDataSet, TorchDataSet]:
        d1, d2 = self.dataSet.split(fractionalSizeOfFirstSet)
        return self._createDataSet(d1), self._createDataSet(d2)

    def _createDataSet(self, d: torchtext.data.Dataset):
        return TorchDataSetFromTorchtextDataSet(d, self.inputField, self.outputField, self.cuda)