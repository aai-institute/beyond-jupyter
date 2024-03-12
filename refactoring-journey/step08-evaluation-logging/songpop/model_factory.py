from sensai.data_transformation import DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureCollector
from sensai.sklearn.sklearn_classification import SkLearnLogisticRegressionVectorClassificationModel, \
    SkLearnKNeighborsVectorClassificationModel, SkLearnRandomForestVectorClassificationModel, SkLearnDecisionTreeVectorClassificationModel
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from .data import *
from .features import FeatureName, registry


class ModelFactory:
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
            .with_name("LogisticRegression")

    @classmethod
    def create_random_forest(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnRandomForestVectorClassificationModel(n_estimators=100) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name("RandomForest")

    @classmethod
    def create_knn(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnKNeighborsVectorClassificationModel(n_neighbors=1) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
                fc.create_feature_transformer_normalisation(),
                DFTSkLearnTransformer(MaxAbsScaler())) \
            .with_name("KNeighbors")
