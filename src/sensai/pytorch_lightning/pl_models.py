import numpy as np
from pytorch_lightning import Trainer, LightningModule
from torch import tensor

from .. import tensor_model as tm
from ..data import InputOutputArrays, DataSplitterFractional


def _fitModelWithTrainer(model: LightningModule, trainer: Trainer, inputOutputData,
                         batchSize: int, splitter: DataSplitterFractional = None):
    if splitter is not None:
        trainIOData, validationIOData = splitter.split(inputOutputData)
        trainDataLoader = trainIOData.toTorchDataLoader(batchSize=batchSize)
        valDataLoader = validationIOData.toTorchDataLoader(batchSize=batchSize)
    else:
        trainDataLoader = inputOutputData.toTorchDataLoader(batchSize=batchSize)
        valDataLoader = None
    trainer.fit(model, trainDataLoader, val_dataloaders=valDataLoader)


class PLWrappedModel:
    def __init__(self, model: LightningModule, trainer: Trainer, validationFraction=0.1, shuffle=True, batchSize=32):
        if not 0 <= validationFraction <= 1:
            raise ValueError(f"Invalid validationFraction: {validationFraction}. Has to be in interval [0, 1]")
        self.trainer = trainer
        self.model = model
        self.validationFraction = validationFraction
        self.shuffle = shuffle
        self.batchSize = batchSize

    def fit(self, X: np.ndarray, Y: np.ndarray):
        inputOutputData = InputOutputArrays(X, Y)
        splitter = DataSplitterFractional(1 - self.validationFraction, shuffle=self.shuffle)
        _fitModelWithTrainer(self.model, self.trainer, inputOutputData, self.batchSize, splitter=splitter)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = tensor(X)
        return self.model(X).detach().cpu().numpy()


# noinspection DuplicatedCode
class PLTensorToScalarClassificationModel(tm.TensorToScalarClassificationModel):
    def __init__(self, model: LightningModule, trainer: Trainer, validationFraction=0.1, shuffle=True, batchSize=64,
                 checkInputShape=True, checkInputColumns=True):
        super().__init__(checkInputShape=checkInputShape, checkInputColumns=checkInputColumns)
        self.wrappedModel = PLWrappedModel(model, trainer, validationFraction=validationFraction, shuffle=shuffle,
                                           batchSize=batchSize)

    def _predictProbabilitiesArray(self, X: np.ndarray) -> np.ndarray:
        return self.wrappedModel.predict(X)

    def _fitToArray(self, X: np.ndarray, Y: np.ndarray):
        self.wrappedModel.fit(X, Y)


# noinspection DuplicatedCode
class PLTensorToScalarRegressionModel(tm.TensorToScalarRegressionModel):
    def __init__(self, model: LightningModule, trainer: Trainer,  validationFraction=0.1, shuffle=True, batchSize=32,
                 checkInputShape=True, checkInputColumns=True):
        super().__init__(checkInputShape=checkInputShape, checkInputColumns=checkInputColumns)
        self.wrappedModel = PLWrappedModel(model, trainer, validationFraction=validationFraction, shuffle=shuffle,
                                           batchSize=batchSize)

    def _predictArray(self, X: np.ndarray) -> np.ndarray:
        return self.wrappedModel.predict(X)

    def _fitToArray(self, X: np.ndarray, Y: np.ndarray):
        self.wrappedModel.fit(X, Y)


# noinspection DuplicatedCode
class PLTensorToTensorClassificationModel(tm.TensorToTensorClassificationModel):
    def __init__(self, model: LightningModule, trainer: Trainer,  validationFraction=0.1, shuffle=True, batchSize=32,
                 checkInputShape=True, checkInputColumns=True):
        super().__init__(checkInputShape=checkInputShape, checkInputColumns=checkInputColumns)
        self.wrappedModel = PLWrappedModel(model, trainer, validationFraction=validationFraction, shuffle=shuffle,
                                           batchSize=batchSize)

    def _predictProbabilitiesArray(self, X: np.ndarray) -> np.ndarray:
        return self.wrappedModel.predict(X)

    def _fitToArray(self, X: np.ndarray, Y: np.ndarray):
        self.wrappedModel.fit(X, Y)


# noinspection DuplicatedCode
class PLTensorToTensorRegressionModel(tm.TensorToTensorRegressionModel):
    def __init__(self, model: LightningModule, trainer: Trainer,  validationFraction=0.1, shuffle=True, batchSize=32,
                 checkInputShape=True, checkInputColumns=True):
        super().__init__(checkInputShape=checkInputShape, checkInputColumns=checkInputColumns)
        self.wrappedModel = PLWrappedModel(model, trainer, validationFraction=validationFraction, shuffle=shuffle,
                                           batchSize=batchSize)

    def _predictArray(self, X: np.ndarray) -> np.ndarray:
        return self.wrappedModel.predict(X)

    def _fitToArray(self, X: np.ndarray, Y: np.ndarray):
        self.wrappedModel.fit(X, Y)
