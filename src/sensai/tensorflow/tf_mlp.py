from tensorflow import keras

from .tf_base import KerasVectorRegressionModel
from .. import normalisation


class KerasMultiLayerPerceptronVectorRegressionModel(KerasVectorRegressionModel):
    def __init__(self, hiddenDims=(5,5), hiddenActivation="sigmoid", outputActivation="sigmoid", loss="mse",
            metrics=("mse",), optimiser="adam", normalisationMode=normalisation.NormalisationMode.MAX_BY_COLUMN, **kwargs):
        super().__init__(normalisationMode, loss, metrics, optimiser, **kwargs)
        self.hiddenDims = hiddenDims
        self.hiddenActivation = hiddenActivation
        self.outputActivation = outputActivation

    def __str__(self):
        params = dict(hiddenDims=self.hiddenDims, hiddenActivation=self.hiddenActivation,
            outputActivation=self.outputActivation)
        return f"{self.__class__.__name__}{params}={super().__str__()}"

    def _createModel(self, inputDim, outputDim):
        modelInputs = keras.Input(shape=(inputDim,), name='input')
        x = modelInputs
        for i, hiddenDim in enumerate(self.hiddenDims):
            x = keras.layers.Dense(hiddenDim, activation=self.hiddenActivation, name='dense_%d' % i)(x)
        modelOutputs = keras.layers.Dense(outputDim, activation=self.outputActivation, name='predictions')(x)
        return keras.Model(inputs=modelInputs, outputs=modelOutputs)


