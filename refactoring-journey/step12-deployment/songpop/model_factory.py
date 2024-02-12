from pathlib import Path
from typing import Sequence

from sensai import VectorRegressionModel
from sensai.data_transformation import DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureCollector
from sensai.sklearn.sklearn_classification import SkLearnLogisticRegressionVectorClassificationModel, \
    SkLearnKNeighborsVectorClassificationModel, SkLearnRandomForestVectorClassificationModel, SkLearnDecisionTreeVectorClassificationModel
from sensai.sklearn.sklearn_regression import SkLearnRandomForestVectorRegressionModel, SkLearnLinearRegressionVectorRegressionModel
from sensai.util.pickle import load_pickle
from sensai.vector_model import RuleBasedVectorClassificationModel
from sensai.xgboost import XGBGradientBoostedVectorClassificationModel, XGBGradientBoostedVectorRegressionModel
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from .data import *
from .features import FeatureName, registry


def best_regression_model_storage_path(dataset: Dataset) -> Path:
    return Path("results") / "models" / "regression" / dataset.tag() / "best_model.pickle"


class VectorClassificationModelFromVectorRegressionModel(RuleBasedVectorClassificationModel):
    def __init__(self, model: VectorRegressionModel, threshold: float, value_lt: str, value_gte: str):
        super().__init__([value_lt, value_gte], predicted_variable_name=COL_GEN_POPULARITY_CLASS)
        self.value_gte = value_gte
        self.value_lt = value_lt
        self.threshold = threshold
        self.model = model

    def _predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        regressor_results = self.model.predict(x).iloc[:, 0]
        rows = []
        for result in regressor_results:
            p_popular = 1 if result >= self.threshold else 0
            rows.append((p_popular, 1 - p_popular))
        return pd.DataFrame(rows, index=x.index, columns=[self.value_gte, self.value_lt])

    def get_name(self):
        return f"FromRegressor[{self.model.get_name()}]"


class ClassificationModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS,
        COL_DURATION_MS]
    DEFAULT_FEATURES = (FeatureName.MUSICAL_DEGREES, FeatureName.MUSICAL_CATEGORIES, FeatureName.TEMPO, FeatureName.DURATION,
        FeatureName.LOUDNESS, FeatureName.YEAR)

    @classmethod
    def create_logistic_regression_orig(cls):
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("LogisticRegression-orig")

    @classmethod
    def create_knn_orig(cls):
        return SkLearnKNeighborsVectorClassificationModel(n_neighbors=1) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("KNeighbors-orig")

    @classmethod
    def create_random_forest_orig(cls):
        return SkLearnRandomForestVectorClassificationModel(n_estimators=100) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("RandomForest-orig")

    @classmethod
    def create_decision_tree_orig(cls):
        return SkLearnDecisionTreeVectorClassificationModel(max_depth=2) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("DecisionTree-orig")

    @classmethod
    def create_logistic_regression(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
                fc.create_feature_transformer_normalisation()) \
            .with_name(f"LogisticRegression")

    @classmethod
    def create_random_forest(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnRandomForestVectorClassificationModel(n_estimators=100) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"RandomForest")

    @classmethod
    def create_knn(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnKNeighborsVectorClassificationModel(n_neighbors=1) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
                fc.create_feature_transformer_normalisation(),
                DFTSkLearnTransformer(MaxAbsScaler())) \
            .with_name("KNeighbors")

    @classmethod
    def create_xgb(cls, name_suffix="", features: Sequence[FeatureName] = DEFAULT_FEATURES, add_features: Sequence[FeatureName] = (),
            min_child_weight: Optional[float] = None, **kwargs):
        fc = FeatureCollector(*features, *add_features, registry=registry)
        return XGBGradientBoostedVectorClassificationModel(min_child_weight=min_child_weight, **kwargs) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"XGBoost{name_suffix}")

    @classmethod
    def create_classifier_from_best_regressor(cls, dataset):
        path = best_regression_model_storage_path(dataset)
        if not path.exists():
            return None
        regression_model = load_pickle(path)
        return VectorClassificationModelFromVectorRegressionModel(regression_model, threshold=dataset.threshold_popular,
            value_lt=dataset.class_negative, value_gte=dataset.class_positive)


class RegressionModelFactory:
    BASE_FEATURES = (FeatureName.MUSICAL_DEGREES, FeatureName.MUSICAL_CATEGORIES, FeatureName.LOUDNESS,
        FeatureName.DURATION, FeatureName.TEMPO, FeatureName.YEAR)

    @classmethod
    def create_rf(cls, name_suffix="", min_samples_leaf=1, **kwargs):
        fc = registry.collect_features(*cls.BASE_FEATURES)
        return SkLearnRandomForestVectorRegressionModel(min_samples_leaf=min_samples_leaf, **kwargs) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"RandomForest{name_suffix}")

    @classmethod
    def create_linear(cls):
        fc = registry.collect_features(*cls.BASE_FEATURES)
        return SkLearnLinearRegressionVectorRegressionModel() \
            .with_feature_collector(fc) \
            .with_feature_transformers(
                fc.create_feature_transformer_one_hot_encoder(ignore_unknown=True),
                fc.create_feature_transformer_normalisation()) \
            .with_name("Linear")

    @classmethod
    def create_xgb(cls, name_suffix="", features: Sequence[FeatureName] = BASE_FEATURES, add_features: Sequence[FeatureName] = (),
            min_child_weight: Optional[float] = None, **kwargs):
        fc = FeatureCollector(*features, *add_features, registry=registry)
        return XGBGradientBoostedVectorRegressionModel(min_child_weight=min_child_weight, **kwargs) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"XGBoost{name_suffix}")

    @classmethod
    def create_xgb_meanpop_opt(cls):
        params = {'colsample_bytree': 0.9869550725977663,
                  'gamma': 8.022497033174522,
                  'max_depth': 10,
                  'min_child_weight': 48.0,
                  'reg_lambda': 0.3984639652186364}
        return cls.create_xgb("-meanPop-opt", add_features=[FeatureName.MEAN_ARTIST_POPULARITY], **params)
