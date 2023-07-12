import logging
from typing import Optional, Sequence

import pandas as pd

from pop.features import COL_POPULARITY, FeatureName, registry
from sensai import InputOutputData
from sensai.data_transformation import DFTOneHotEncoder, DFTNormalisation, DFTKeepColumns
from sensai.evaluation import VectorRegressionModelEvaluatorParams, RegressionEvaluationUtil, \
    VectorModelCrossValidatorParams
from sensai.featuregen import FeatureCollector
from sensai.sklearn.sklearn_regression import SkLearnRandomForestVectorRegressionModel, \
    SkLearnLinearRegressionVectorRegressionModel
from sensai.util.string import ToStringMixin
from sensai.xgboost import XGBGradientBoostedVectorRegressionModel, XGBRandomForestVectorRegressionModel

log = logging.getLogger(__name__)


class DataSet(ToStringMixin):
    def __init__(self, num_samples: Optional[int], random_state=42):
        self.num_samples = num_samples
        self.random_state = random_state

    def tag(self) -> str:
        return f"numSamples{self.num_samples}rnd{self.random_state}"

    def load_iodata(self) -> InputOutputData:
        log.info(f"Reading data (num_samples={self.num_samples})")
        df = pd.read_csv("data/spotify_data.csv").dropna()
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_state)
        return InputOutputData.fromDataFrame(df, COL_POPULARITY)


class ModelFactory:
    BASE_FEATURES = [FeatureName.MUSICAL_DEGREES, FeatureName.MUSICAL_CATEGORIES, FeatureName.LOUDNESS,
        FeatureName.DURATION, FeatureName.TEMPO]

    @classmethod
    def _feature_collector(cls, add_features=()):
        return FeatureCollector(*cls.BASE_FEATURES, *add_features, registry=registry)

    @classmethod
    def create_rf(cls, name_suffix="", min_samples_leaf=1, **kwargs):
        fc = cls._feature_collector()
        return SkLearnRandomForestVectorRegressionModel(min_samples_leaf=min_samples_leaf, **kwargs) \
            .withFeatureCollector(fc) \
            .withFeatureTransformers(DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex())) \
            .withName(f"RandomForest{name_suffix}")

    @classmethod
    def create_linear(cls):
        fc = cls._feature_collector()
        return SkLearnLinearRegressionVectorRegressionModel() \
            .withFeatureCollector(fc) \
            .withFeatureTransformers(DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex()),
                DFTNormalisation(fc.getNormalisationRules())) \
            .withName("Linear")

    @classmethod
    def create_xgb(cls, name_suffix="", add_features: Sequence[FeatureName] = (), **kwargs):
        fc = cls._feature_collector(add_features)
        return XGBGradientBoostedVectorRegressionModel(**kwargs) \
            .withFeatureCollector(fc) \
            .withFeatureTransformers(DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex())) \
            .withName(f"XGBoost{name_suffix}")

    @classmethod
    def create_xgb_meanpop_opt(cls):
        params = {'colsample_bytree': 0.5405227696378077, 'gamma': 5.501809177366219, 'max_depth': 4,
            'min_child_weight': 12, 'reg_lambda': 0.3379373357937709}
        return cls.create_xgb(name_suffix="-meanPop-opt", add_features=[FeatureName.MEAN_ARTIST_POPULARITY],
            **params)

    @classmethod
    def create_xgb_meanpop_opt_fsel(cls):
        model = cls.create_xgb_meanpop_opt()
        selected_feature_columns = ['mean_artist_popularity', 'genre_2', 'genre_10', 'genre_13', 'genre_14',
            'genre_24', 'genre_27', 'genre_35', 'genre_42', 'genre_47', 'genre_49', 'genre_56']
        return model.withFeatureTransformers(DFTKeepColumns(selected_feature_columns), add=True)

    @classmethod
    def create_xgbrf(cls, name_suffix="", add_features: Sequence[FeatureName] = ()):
        fc = cls._feature_collector(add_features)
        return XGBRandomForestVectorRegressionModel() \
            .withFeatureCollector(fc) \
            .withFeatureTransformers(DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex())) \
            .withName(f"XGBRandomForest{name_suffix}")


class ModelEvaluation:
    def __init__(self, dataset: DataSet):
        self.dataset = dataset
        self.iodata = dataset.load_iodata()
        self.evaluator_params = VectorRegressionModelEvaluatorParams(fractionalSplitTestFraction=0.2,
            fractionalSplitShuffle=True)
        self.crossval_params = VectorModelCrossValidatorParams(folds=5, evaluatorParams=self.evaluator_params)

    def create_evaluator(self):
        return RegressionEvaluationUtil(self.iodata, evaluatorParams=self.evaluator_params,
            crossValidatorParams=self.crossval_params)
