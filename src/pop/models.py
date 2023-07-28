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
        return f"numSamples{self.num_samples}Rnd{self.random_state}"

    def load_iodata(self) -> InputOutputData:
        log.info(f"Reading data (num_samples={self.num_samples})")
        df = pd.read_csv("data/spotify_data.csv").dropna()
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_state)
        return InputOutputData.from_data_frame(df, COL_POPULARITY)


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
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"RandomForest{name_suffix}")

    @classmethod
    def create_linear(cls):
        fc = cls._feature_collector()
        return SkLearnLinearRegressionVectorRegressionModel() \
            .with_feature_collector(fc) \
            .with_feature_transformers(
                fc.create_feature_transformer_one_hot_encoder(ignore_unknown=True),
                fc.create_feature_transformer_normalisation()) \
            .with_name("Linear")

    @classmethod
    def create_xgb(cls, name_suffix="", add_features: Sequence[FeatureName] = (), **kwargs):
        fc = cls._feature_collector(add_features)
        return XGBGradientBoostedVectorRegressionModel(**kwargs) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"XGBoost{name_suffix}")

    @classmethod
    def create_xgb_meanpop_opt(cls):
        params = {'colsample_bytree': 0.5405227696378077, 'gamma': 5.501809177366219, 'max_depth': 4,
            'min_child_weight': 12, 'reg_lambda': 0.3379373357937709}
        return cls.create_xgb(name_suffix="-meanPop-opt", add_features=[FeatureName.MEAN_ARTIST_POPULARITY],
            **params)

    @classmethod
    def create_xgb_meanpop_opt_fsel(cls):
        model = cls.create_xgb_meanpop_opt()
        model.set_name(model.get_name() + "-fsel")
        selected_feature_columns = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'loudness', 'duration_ms', 'mean_artist_popularity', 'genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_6', 'genre_7', 'genre_8', 'genre_9', 'genre_10', 'genre_11', 'genre_12', 'genre_13', 'genre_14', 'genre_15', 'genre_16', 'genre_17', 'genre_18', 'genre_22', 'genre_23', 'genre_24', 'genre_25', 'genre_26', 'genre_27', 'genre_28', 'genre_29', 'genre_30', 'genre_32', 'genre_33', 'genre_34', 'genre_35', 'genre_36', 'genre_37', 'genre_39', 'genre_40', 'genre_41', 'genre_42', 'genre_43', 'genre_44', 'genre_45', 'genre_46', 'genre_47', 'genre_48', 'genre_49', 'genre_51', 'genre_52', 'genre_53', 'genre_54', 'genre_56', 'genre_57', 'genre_58', 'genre_61', 'genre_63', 'genre_64', 'genre_65', 'genre_66', 'genre_67', 'genre_68', 'genre_69', 'genre_70', 'genre_72', 'genre_73', 'genre_75', 'genre_76', 'genre_78', 'genre_81', 'mode_0']
        return model.with_feature_transformers(DFTKeepColumns(selected_feature_columns), add=True)

    @classmethod
    def create_xgbrf(cls, name_suffix="", add_features: Sequence[FeatureName] = ()):
        fc = cls._feature_collector(add_features)
        return XGBRandomForestVectorRegressionModel() \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"XGBRandomForest{name_suffix}")


class ModelEvaluation:
    def __init__(self, dataset: DataSet):
        self.dataset = dataset
        self.iodata = dataset.load_iodata()
        self.evaluator_params = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.2,
            fractional_split_shuffle=True)
        self.crossval_params = VectorModelCrossValidatorParams(folds=5, evaluator_params=self.evaluator_params)

    def create_evaluator(self):
        return RegressionEvaluationUtil(self.iodata, evaluator_params=self.evaluator_params,
            cross_validator_params=self.crossval_params)
