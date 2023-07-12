import logging
from copy import copy
from dataclasses import dataclass
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np

from sensai import VectorModel, InputOutputData
from sensai.evaluation import VectorModelCrossValidatorParams, createVectorModelCrossValidator
from sensai.feature_importance import FeatureImportanceProvider, AggregatedFeatureImportance
from sensai.util.plot import ScatterPlot

log = logging.getLogger(__name__)


class RecursiveFeatureEliminationCV:
    """
    Recursive feature elimination, using cross-validation to select the best set of features:
    In each step, the model is first evaluated using cross-validation.
    Then the feature importance values are aggregated across the models that were trained during cross-validation,
    and the least important feature is discarded. For the case where the lowest feature importance is 0, all
    features with 0 importance are discarded.
    This process is repeated until a point is reached where only `minFeatures` (or less) remain.
    The selected set of features is the one from the step where cross-validation yielded the best evaluation metric value.

    Feature importance is computed at the level of model input features, i.e. after feature generation and transformation.

    NOTE: This implementation differs markedly from sklearn's RFECV, which performs an independent RFE for each fold.
    RFECV determines the number of features to use by determining the elimination step in each fold that yielded the best
    metric value on average. Because the eliminations are independent, the actual features that were being used in those step
    could have been completely different. Using the selected number of features n, RFECV then performs another RFE, eliminating features
    until n features remain and returns these features as the result.
    """
    def __init__(self, crossValidatorParams: VectorModelCrossValidatorParams, minFeatures=1):
        """
        :param crossValidatorParams: the parameters for cross-validation
        :param minFeatures: the minimum number of features to evaluate
        """
        self.crossValidatorParams = crossValidatorParams
        self.minFeatures = minFeatures

    @dataclass
    class Step:
        metricValue: float
        features: List[str]

    class Result:
        def __init__(self, steps: List["RecursiveFeatureEliminationCV.Step"], metricName: str, minimise: bool):
            self.steps = steps
            self.metricName = metricName
            self.minimise = minimise

        def getSortedSteps(self) -> List["RecursiveFeatureEliminationCV.Step"]:
            """
            :return: the elimination step results, sorted from best to worst
            """
            return sorted(self.steps, key=lambda s: s.metricValue, reverse=not self.minimise)

        def getSelectedFeatures(self) -> List[str]:
            return self.getSortedSteps()[0].features

        def getNumFeaturesArray(self) -> np.ndarray:
            """
            :return: array containing the number of features that was considered in each step
            """
            return np.array([len(s.features) for s in self.steps])

        def getMetricValuesArray(self) -> np.ndarray:
            """
            :return: array containing the metric value that resulted in each step
            """
            return np.array([s.metricValue for s in self.steps])

        def plotMetricValues(self) -> plt.Figure:
            """
            Plots the metric values vs. the number of features for each step of the elimination

            :return: the figure
            """
            return ScatterPlot(self.getNumFeaturesArray(), self.getMetricValuesArray(), c_opacity=1, x_label="number of features",
                y_label=f"cross-validation mean metric value ({self.metricName})").fig

    def run(self, model: Union[VectorModel, FeatureImportanceProvider], ioData: InputOutputData, metricName: str, minimise: bool) -> Result:
        """
        Runs the optimisation for the given model and data.

        :param model: the model
        :param ioData: the data
        :param metricName: the metric to optimise
        :param minimise: whether the metric shall be minimsed; if False, maximise.
        :return: a result object, which provides access to the selected features and data on all elimination steps
        """
        metricKey = f"mean[{metricName}]"

        model = copy(model)
        model.fitInputOutputData(ioData, fitPreprocessors=True, fitModel=False)
        inputs = model.computeModelInputs(ioData.inputs)
        model.removeInputPreprocessors()
        ioData = InputOutputData(inputs, ioData.outputs)

        features = list(inputs.columns)
        steps = []
        while True:
            # evaluate model
            crossValidator = createVectorModelCrossValidator(ioData, model=model, params=self.crossValidatorParams)
            crossValData = crossValidator.evalModel(model)
            aggMetricsDict = crossValData.getEvalStatsCollection().aggMetricsDict()
            metricValue = aggMetricsDict[metricKey]

            steps.append(self.Step(metricValue=metricValue, features=features))

            # eliminate feature(s)
            log.info(f"Model performance with {len(features)} features: {metricKey}={metricValue}")
            aggImportance = AggregatedFeatureImportance(*crossValData.trainedModels)
            fi = aggImportance.getAggregatedFeatureImportance()
            tuples = fi.getSortedTuples()
            minImportance = tuples[0][1]
            if minImportance == 0:
                eliminatedFeatures = []
                for i, (fname, importance) in enumerate(tuples):
                    if importance > 0:
                        break
                    eliminatedFeatures.append(fname)
                log.info(f"Eliminating {len(eliminatedFeatures)} features with 0 importance: {eliminatedFeatures}")
            else:
                eliminatedFeatures = [tuples[0][0]]
                log.info(f"Eliminating feature {eliminatedFeatures[0]}")
            features = [f for f in features if f not in eliminatedFeatures]
            ioData.inputs = ioData.inputs[features]
            log.info(f"{len(features)} features remain")

            if len(features) < self.minFeatures:
                log.info("Minimum number of features reached/exceeded")
                break

        return self.Result(steps, metricName, minimise)