import numpy as np
from pytorch_lightning import Trainer, LightningModule
from torch import tensor

from .. import tensor_model as tm
from ..data import InputOutputArrays, DataSplitterFractional


def _fit_model_with_trainer(model: LightningModule, trainer: Trainer, io_data,
        batch_size: int, splitter: DataSplitterFractional = None):
    if splitter is not None:
        train_io_data, validation_io_data = splitter.split(io_data)
        train_data_loader = train_io_data.toTorchDataLoader(batchSize=batch_size)
        val_data_loader = validation_io_data.toTorchDataLoader(batchSize=batch_size)
    else:
        train_data_loader = io_data.to_torch_data_loader(batch_size=batch_size)
        val_data_loader = None
    trainer.fit(model, train_data_loader, val_dataloaders=val_data_loader)


class PLWrappedModel:
    def __init__(self, model: LightningModule, trainer: Trainer, validation_fraction=0.1, shuffle=True, batch_size=32):
        if not 0 <= validation_fraction <= 1:
            raise ValueError(f"Invalid validationFraction: {validation_fraction}. Has to be in interval [0, 1]")
        self.trainer = trainer
        self.model = model
        self.validationFraction = validation_fraction
        self.shuffle = shuffle
        self.batchSize = batch_size

    def fit(self, x: np.ndarray, y: np.ndarray):
        io_data = InputOutputArrays(x, y)
        splitter = DataSplitterFractional(1 - self.validationFraction, shuffle=self.shuffle)
        _fit_model_with_trainer(self.model, self.trainer, io_data, self.batchSize, splitter=splitter)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = tensor(x)
        return self.model(x).detach().cpu().numpy()


# noinspection DuplicatedCode
class PLTensorToScalarClassificationModel(tm.TensorToScalarClassificationModel):
    def __init__(self, model: LightningModule, trainer: Trainer, validation_fraction=0.1, shuffle=True, batch_size=64,
                 check_input_shape=True, check_input_columns=True):
        super().__init__(check_input_shape=check_input_shape, check_input_columns=check_input_columns)
        self.wrapped_model = PLWrappedModel(model, trainer, validation_fraction=validation_fraction, shuffle=shuffle,
                                           batch_size=batch_size)

    def _predict_probabilities_array(self, x: np.ndarray) -> np.ndarray:
        return self.wrapped_model.predict(x)

    def _fit_to_array(self, x: np.ndarray, y: np.ndarray):
        self.wrapped_model.fit(x, y)


# noinspection DuplicatedCode
class PLTensorToScalarRegressionModel(tm.TensorToScalarRegressionModel):
    def __init__(self, model: LightningModule, trainer: Trainer, validation_fraction=0.1, shuffle=True, batch_size=32,
                 check_input_shape=True, check_input_columns=True):
        super().__init__(check_input_shape=check_input_shape, check_input_columns=check_input_columns)
        self.wrapped_model = PLWrappedModel(model, trainer, validation_fraction=validation_fraction, shuffle=shuffle,
                                           batch_size=batch_size)

    def _predict_array(self, x: np.ndarray) -> np.ndarray:
        return self.wrapped_model.predict(x)

    def _fit_to_array(self, x: np.ndarray, y: np.ndarray):
        self.wrapped_model.fit(x, y)


# noinspection DuplicatedCode
class PLTensorToTensorClassificationModel(tm.TensorToTensorClassificationModel):
    def __init__(self, model: LightningModule, trainer: Trainer, validation_fraction=0.1, shuffle=True, batch_size=32,
                 check_input_shape=True, check_input_columns=True):
        super().__init__(check_input_shape=check_input_shape, check_input_columns=check_input_columns)
        self.wrapped_model = PLWrappedModel(model, trainer, validation_fraction=validation_fraction, shuffle=shuffle,
                                           batch_size=batch_size)

    def _predict_probabilities_array(self, x: np.ndarray) -> np.ndarray:
        return self.wrapped_model.predict(x)

    def _fit_to_array(self, x: np.ndarray, y: np.ndarray):
        self.wrapped_model.fit(x, y)


# noinspection DuplicatedCode
class PLTensorToTensorRegressionModel(tm.TensorToTensorRegressionModel):
    def __init__(self, model: LightningModule, trainer: Trainer, validation_fraction=0.1, shuffle=True, batch_size=32,
                 check_input_shape=True, check_input_columns=True):
        super().__init__(check_input_shape=check_input_shape, check_input_columns=check_input_columns)
        self.wrapped_model = PLWrappedModel(model, trainer, validation_fraction=validation_fraction, shuffle=shuffle,
                                           batch_size=batch_size)

    def _predict_array(self, x: np.ndarray) -> np.ndarray:
        return self.wrapped_model.predict(x)

    def _fit_to_array(self, x: np.ndarray, y: np.ndarray):
        self.wrapped_model.fit(x, y)
