import collections
from math import log, exp

import numpy as np
import pandas as pd

from .vector_model import VectorClassificationModel


class CategoricalNaiveBayesVectorClassificationModel(VectorClassificationModel):
    """
    Naive Bayes with categorical features
    """
    def __init__(self, pseudoCount=0.1):
        """
        :param pseudoCount: the count to add to each empirical count in order to avoid overfitting
        """
        super().__init__()
        self.prior = None
        self.conditionals = None
        self.pseudoCount = pseudoCount

    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        self.prior = collections.defaultdict(lambda: 0)
        self.conditionals = collections.defaultdict(lambda: [collections.defaultdict(lambda: 0) for _ in range(X.shape[1])])
        increment = 1
        for idxRow in range(X.shape[0]):
            cls = y.iloc[idxRow,0]
            self.prior[cls] += increment
            for idxFeature in range(X.shape[1]):
                value = X.iloc[idxRow, idxFeature]
                self.conditionals[cls][idxFeature][value] += increment
        # get rid of defaultdicts, which are not picklable
        self.prior = dict(self.prior)
        self.conditionals = {k: [dict(d) for d in l] for k, l in self.conditionals.items()}

    def _predictClassProbabilities(self, X: pd.DataFrame):
        results = []
        for _, features in X.iterrows():
            classProbabilities = np.zeros(len(self._labels))
            for i, cls in enumerate(self._labels):
                lp = log(self._probability(self.prior, cls))
                for idxFeature, value in enumerate(features):
                    lp += log(self._probability(self.conditionals[cls][idxFeature], value))
                classProbabilities[i] = exp(lp)
            classProbabilities /= np.sum(classProbabilities)
            results.append(classProbabilities)
        return pd.DataFrame(results, columns=self._labels)

    def _probability(self, counts, value):
        valueCount = counts.get(value, 0.0)
        totalCount = sum(counts.values())
        return (valueCount + self.pseudoCount) / (totalCount + self.pseudoCount)

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, features in X.iterrows():
            bestCls = None
            bestLp = None
            for cls in self.prior:
                lp = log(self._probability(self.prior, cls))
                for idxFeature, value in enumerate(features):
                    lp += log(self._probability(self.conditionals[cls][idxFeature], value))
                if bestLp is None or lp > bestLp:
                    bestLp = lp
                    bestCls = cls
            results.append(bestCls)
        return pd.DataFrame(results, columns=self.getPredictedVariableNames())
