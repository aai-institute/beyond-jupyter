from sensai.util import logging
from typing import Optional, Tuple

import pandas as pd

from sensai import InputOutputData
from sensai.util.string import ToStringMixin, TagBuilder
from sklearn.preprocessing import StandardScaler

from . import config


log = logging.getLogger(__name__)


COL_POPULARITY = "popularity"
COL_GEN_POPULARITY_CLASS = "popularity_class"
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

CLASS_POPULAR = "popular"
CLASS_UNPOPULAR = "unpopular"


class Dataset(ToStringMixin):
    def __init__(self, num_samples: Optional[int] = None, drop_zero_popularity: bool = False, threshold_popular: int = 50,
            random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param drop_zero_popularity: whether to drop data points where the popularity is zero
        :param threshold_popular: the threshold below which a song is considered as unpopular
        :param random_seed: the random seed to use when sampling data points
        """
        self.num_samples = num_samples
        self.threshold_popular = threshold_popular
        self.drop_zero_popularity = drop_zero_popularity
        self.random_seed = random_seed
        self.class_positive = self.class_negative = None
        self.col_target = COL_POPULARITY

    def tag(self):
        return TagBuilder(glue="-") \
            .with_alternative(self.num_samples is None, "full", f"numSamples{self.num_samples}") \
            .with_conditional(self.drop_zero_popularity, "drop") \
            .with_conditional(self.threshold_popular != 50, f"threshold{self.threshold_popular}") \
            .with_conditional(self.random_seed != 42, f"seed{self.random_seed}") \
            .build()

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        csv_path = config.csv_data_path()
        log.info(f"Loading {self} from {csv_path}")
        df = pd.read_csv(csv_path).dropna()
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        return df

    def load_io_data(self) -> InputOutputData:
        """
        :return: the I/O data
        """
        return InputOutputData.from_data_frame(self.load_data_frame(), self.col_target)

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        return df.drop(columns=COL_POPULARITY), df[COL_POPULARITY]

    def load_xy_projected_scaled(self) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = self.load_xy()
        cols_used_by_models = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS, COL_DURATION_MS]
        X_proj = X[cols_used_by_models]
        scaler = StandardScaler()
        scaler.fit(X_proj)
        X_scaled = scaler.transform(X_proj)
        return X_scaled, y
