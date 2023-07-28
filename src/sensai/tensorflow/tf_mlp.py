from tensorflow import keras

from .tf_base import KerasVectorRegressionModel
from .. import normalisation


class KerasMultiLayerPerceptronVectorRegressionModel(KerasVectorRegressionModel):
    def __init__(self, hidden_dims=(5,5), hidden_activation="sigmoid", output_activation="sigmoid", loss="mse",
            metrics=("mse",), optimiser="adam", normalisation_mode=normalisation.NormalisationMode.MAX_BY_COLUMN, **kwargs):
        super().__init__(normalisation_mode, loss, metrics, optimiser, **kwargs)
        self.hiddenDims = hidden_dims
        self.hiddenActivation = hidden_activation
        self.outputActivation = output_activation

    def __str__(self):
        params = dict(hiddenDims=self.hiddenDims, hiddenActivation=self.hiddenActivation,
            outputActivation=self.outputActivation)
        return f"{self.__class__.__name__}{params}={super().__str__()}"

    def _create_model(self, input_dim, output_dim):
        model_inputs = keras.Input(shape=(input_dim,), name='input')
        x = model_inputs
        for i, hiddenDim in enumerate(self.hiddenDims):
            x = keras.layers.Dense(hiddenDim, activation=self.hiddenActivation, name='dense_%d' % i)(x)
        model_outputs = keras.layers.Dense(output_dim, activation=self.outputActivation, name='predictions')(x)
        return keras.Model(inputs=model_inputs, outputs=model_outputs)


