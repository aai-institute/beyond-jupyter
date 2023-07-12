from enum import Enum

import numpy as np
import pandas as pd

from sensai import VectorRegressionModel
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns, FeatureGenerator

COL_POPULARITY = "popularity"
# identifying meta-data
COL_ARTIST_NAME = "artist_name"
COL_TRACK_NAME = "track_name"
COL_TRACK_ID = "track_id"
# categorical features
COL_GENRE = "genre"
COL_KEY = "key"
COL_MODE = "mode"  # binary
COLS_MUSICAL_CATEGORIES = [COL_GENRE, COL_KEY, COL_MODE]
# normalised numeric features
COL_DANCEABILITY = "danceability"
COL_ENERGY = "energy"
COL_SPEECHINESS = "speechiness"
COL_ACOUSTICNESS = "acousticness"
COL_INSTRUMENTALNESS = "instrumentalness"
COL_LIVENESS = "liveness"
COL_VALENCE = "valence"
COLS_MUSICAL_DEGREES = [COL_DANCEABILITY, COL_ENERGY, COL_SPEECHINESS, COL_ACOUSTICNESS, COL_INSTRUMENTALNESS,
    COL_LIVENESS, COL_VALENCE]
# other numeric features
COL_YEAR = "year"
COL_LOUDNESS = "loudness"  # probably RMS or LUFS
COL_TEMPO = "tempo"  # BPM
COL_DURATION_MS = "duration_ms"
COL_TIME_SIGNATURE = "time_signature"  # probably notes per bar (but non-uniform semantics: quarter or eighth notes)


class FeatureName(Enum):
    MUSICAL_DEGREES = "musical_degrees"
    MUSICAL_CATEGORIES = "musical_categories"
    LOUDNESS = "loudness"
    TEMPO = "tempo"
    DURATION = "duration"
    MEAN_ARTIST_POPULARITY = "mean_artist_popularity"


class FeatureGeneratorMeanArtistPopularity(FeatureGenerator):
    def __init__(self):
        super().__init__(normalisationRuleTemplate=DFTNormalisation.RuleTemplate(
            transformerFactory=SkLearnTransformerFactoryFactory.MaxAbsScaler()))

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame = None, ctx=None):
        df: pd.DataFrame = pd.concat([X, Y], axis=1)[[COL_ARTIST_NAME, COL_POPULARITY]]
        gb = df.groupby(COL_ARTIST_NAME)
        s = gb.sum()[COL_POPULARITY]
        s.name = "sum"
        c = gb.count()[COL_POPULARITY]
        c.name = "cnt"
        m = s / c
        m.name = "mean"
        self._y = Y
        self._values = pd.concat([s, c, m], axis=1)

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        ctx: VectorRegressionModel
        is_training = ctx.isBeingFitted()

        if is_training:
            def val_t(t):
                lookup = self._values.loc[getattr(t, COL_ARTIST_NAME)]
                s = lookup["sum"] - self._y.loc[t.Index][COL_POPULARITY]
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
registry.registerFactory(FeatureName.MUSICAL_DEGREES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_DEGREES,
    normalisationRuleTemplate=DFTNormalisation.RuleTemplate(skip=True)))
registry.registerFactory(FeatureName.MUSICAL_CATEGORIES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_CATEGORIES,
    categoricalFeatureNames=COLS_MUSICAL_CATEGORIES))
registry.registerFactory(FeatureName.LOUDNESS, lambda: FeatureGeneratorTakeColumns(COL_LOUDNESS,
    normalisationRuleTemplate=DFTNormalisation.RuleTemplate(
        transformerFactory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.registerFactory(FeatureName.TEMPO, lambda: FeatureGeneratorTakeColumns(COL_TEMPO,
    normalisationRuleTemplate=DFTNormalisation.RuleTemplate(
        transformerFactory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.registerFactory(FeatureName.DURATION, lambda: FeatureGeneratorTakeColumns(COL_DURATION_MS,
    normalisationRuleTemplate=DFTNormalisation.RuleTemplate(
        transformerFactory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.registerFactory(FeatureName.MEAN_ARTIST_POPULARITY, FeatureGeneratorMeanArtistPopularity)

