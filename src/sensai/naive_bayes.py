import collections
from math import log, exp

import numpy as np
import pandas as pd

from .vector_model import VectorClassificationModel


class CategoricalNaiveBayesVectorClassificationModel(VectorClassificationModel):
    """
    Naive Bayes with categorical features
    """
    def __init__(self, pseudo_count=0.1):
        """
        :param pseudo_count: the count to add to each empirical count in order to avoid overfitting
        """
        super().__init__()
        self.prior = None
        self.conditionals = None
        self.pseudoCount = pseudo_count

    def _fit_classifier(self, x: pd.DataFrame, y: pd.DataFrame):
        self.prior = collections.defaultdict(lambda: 0)
        self.conditionals = collections.defaultdict(lambda: [collections.defaultdict(lambda: 0) for _ in range(x.shape[1])])
        increment = 1
        for idxRow in range(x.shape[0]):
            cls = y.iloc[idxRow, 0]
            self.prior[cls] += increment
            for idxFeature in range(x.shape[1]):
                value = x.iloc[idxRow, idxFeature]
                self.conditionals[cls][idxFeature][value] += increment
        # get rid of defaultdicts, which are not picklable
        self.prior = dict(self.prior)
        self.conditionals = {k: [dict(d) for d in l] for k, l in self.conditionals.items()}

    def _predict_class_probabilities(self, x: pd.DataFrame):
        results = []
        for _, features in x.iterrows():
            class_probabilities = np.zeros(len(self._labels))
            for i, cls in enumerate(self._labels):
                lp = log(self._probability(self.prior, cls))
                for idx_feature, value in enumerate(features):
                    lp += log(self._probability(self.conditionals[cls][idx_feature], value))
                class_probabilities[i] = exp(lp)
            class_probabilities /= np.sum(class_probabilities)
            results.append(class_probabilities)
        return pd.DataFrame(results, columns=self._labels)

    def _probability(self, counts, value):
        value_count = counts.get(value, 0.0)
        total_count = sum(counts.values())
        return (value_count + self.pseudoCount) / (total_count + self.pseudoCount)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, features in x.iterrows():
            best_cls = None
            best_lp = None
            for cls in self.prior:
                lp = log(self._probability(self.prior, cls))
                for idxFeature, value in enumerate(features):
                    lp += log(self._probability(self.conditionals[cls][idxFeature], value))
                if best_lp is None or lp > best_lp:
                    best_lp = lp
                    best_cls = cls
            results.append(best_cls)
        return pd.DataFrame(results, columns=self.get_predicted_variable_names())
