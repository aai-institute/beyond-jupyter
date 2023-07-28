import pandas as pd

from ..vector_model import VectorModel


class VectorModelWithSeparateFeatureGeneration:
    def __init__(self, vector_model: VectorModel):
        self.vectorModel = vector_model
        self.featureGen = vector_model.get_feature_generator()
        self.vectorModel.set_feature_generator(None)

    def __str__(self):
        return self.vectorModel.__str__()

    class IntermediateFittingStep:
        def __init__(self, vector_model: VectorModel, x: pd.DataFrame, y: pd.DataFrame):
            self.y = y
            self.x = x
            self.vector_model = vector_model

        def execute(self) -> VectorModel:
            self.vector_model.fit(self.x, self.y)
            return self.vector_model

        def __str__(self):
            return f"{self.__class__.__name__} for {self.vector_model}"

    class PredictFinaliser:
        def __init__(self, vector_model: VectorModel, x: pd.DataFrame):
            self.X = x
            self.vectorModel = vector_model

        def execute(self) -> pd.DataFrame:
            return self.vectorModel.predict(self.X)

        def __str__(self):
            return f"{self.__class__.__name__} for {self.vectorModel}"

    def fit_start(self, x, y) -> 'VectorModelWithSeparateFeatureGeneration.IntermediateFittingStep':
        x = self.featureGen.fit_generate(x, y)
        return self.IntermediateFittingStep(self.vectorModel, x, y)

    def predict_start(self, x: pd.DataFrame):
        x = self.featureGen.generate(x)
        return self.PredictFinaliser(self.vectorModel, x)

    def fit_end(self, vector_model) -> VectorModel:
        vector_model._featureGenerator = self.featureGen
        return vector_model
