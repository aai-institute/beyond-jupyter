from sensai.data_transformation import DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns
from sensai.sklearn.sklearn_classification import SkLearnLogisticRegressionVectorClassificationModel, \
    SkLearnKNeighborsVectorClassificationModel, SkLearnRandomForestVectorClassificationModel, SkLearnDecisionTreeVectorClassificationModel
from sklearn.preprocessing import StandardScaler

from .data import COL_YEAR, COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS, COL_DURATION_MS


class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS,
        COL_DURATION_MS]

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
