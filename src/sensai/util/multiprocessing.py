import pandas as pd

from ..vector_model import VectorModel


class VectorModelWithSeparateFeatureGeneration:
    def __init__(self, vectorModel: VectorModel):
        self.vectorModel = vectorModel
        self.featureGen = vectorModel.getFeatureGenerator()
        self.vectorModel.setFeatureGenerator(None)

    def __str__(self):
        return self.vectorModel.__str__()

    class IntermediateFittingStep:
        def __init__(self, vectorModel: VectorModel, X: pd.DataFrame, Y: pd.DataFrame):
            self.Y = Y
            self.X = X
            self.vectorModel = vectorModel

        def execute(self) -> VectorModel:
            self.vectorModel.fit(self.X, self.Y)
            return self.vectorModel

        def __str__(self):
            return f"{self.__class__.__name__} for {self.vectorModel}"

    class PredictFinaliser:
        def __init__(self, vectorModel: VectorModel, X: pd.DataFrame):
            self.X = X
            self.vectorModel = vectorModel

        def execute(self) -> pd.DataFrame:
            return self.vectorModel.predict(self.X)

        def __str__(self):
            return f"{self.__class__.__name__} for {self.vectorModel}"

    def fitStart(self, X, Y) -> 'VectorModelWithSeparateFeatureGeneration.IntermediateFittingStep':
        X = self.featureGen.fitGenerate(X, Y)
        return self.IntermediateFittingStep(self.vectorModel, X, Y)

    def predictStart(self, X: pd.DataFrame):
        X = self.featureGen.generate(X)
        return self.PredictFinaliser(self.vectorModel, X)

    def fitEnd(self, vectorModel) -> VectorModel:
        vectorModel._featureGenerator = self.featureGen
        return vectorModel
