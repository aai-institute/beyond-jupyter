from enum import Enum

import numpy as np
from sensai import VectorRegressionModel
from .data import *
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns, FeatureGenerator


class FeatureName(Enum):
    MUSICAL_DEGREES = "musical_degrees"
    MUSICAL_CATEGORIES = "musical_categories"
    LOUDNESS = "loudness"
    TEMPO = "tempo"
    DURATION = "duration"
    MEAN_ARTIST_FREQ_POPULAR = "mean_artist_freq_popular"


class FeatureGeneratorMeanArtistPopularity(FeatureGenerator):
    def __init__(self):
        super().__init__(normalisation_rule_template=DFTNormalisation.RuleTemplate(
            transformer_factory=SkLearnTransformerFactoryFactory.MaxAbsScaler()))
        self.col_target = COL_GEN_POPULARITY_CLASS
        self._y = None

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        df: pd.DataFrame = pd.concat([x, y], axis=1)[[COL_ARTIST_NAME, self.col_target]]
        df[self.col_target] = df[self.col_target].apply(lambda x: 1 if x == CLASS_POPULAR else 0)
        gb = df.groupby(COL_ARTIST_NAME)
        s = gb.sum()[self.col_target]
        s.name = "sum"
        c = gb.count()[self.col_target]
        c.name = "cnt"
        m = s / c
        m.name = "mean"
        self._y = df[[self.col_target]]
        self._values = pd.concat([s, c, m], axis=1)

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        ctx: VectorRegressionModel
        is_training = ctx.is_being_fitted()

        if is_training:
            def val_t(t):
                lookup = self._values.loc[getattr(t, COL_ARTIST_NAME)]
                s = lookup["sum"] - self._y.loc[t.Index][self.col_target]
                c = lookup["cnt"] - 1
                if c == 0:
                    return np.nan
                else:
                    return s / c

            values = [val_t(t) for t in df.itertuples()]

            # clean up
            self._y = None
            self._values.drop(columns=["sum", "cnt"])
        else:
            def val_i(artist_name):
                try:
                    return self._values.loc[artist_name]["mean"]
                except KeyError:
                    return np.nan

            values = df[COL_ARTIST_NAME].apply(val_i)

        return pd.DataFrame({"mean_artist_popularity": values}, index=df.index)


registry = FeatureGeneratorRegistry()
registry.register_factory(FeatureName.MUSICAL_DEGREES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_DEGREES,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)))
registry.register_factory(FeatureName.MUSICAL_CATEGORIES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_CATEGORIES,
    categorical_feature_names=COLS_MUSICAL_CATEGORIES))
registry.register_factory(FeatureName.LOUDNESS, lambda: FeatureGeneratorTakeColumns(COL_LOUDNESS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.register_factory(FeatureName.TEMPO, lambda: FeatureGeneratorTakeColumns(COL_TEMPO,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.register_factory(FeatureName.DURATION, lambda: FeatureGeneratorTakeColumns(COL_DURATION_MS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.register_factory(FeatureName.MEAN_ARTIST_FREQ_POPULAR, FeatureGeneratorMeanArtistPopularity)
