# Step 2: Dataset Representation

In this step, we introduce a representation for the data set:
 * **We explicitly represent the parameters that determine the data** in the attributes 
   of a newly introduced class `Dataset`, making our choices explicit: 
   * We don't really agree with the notion of dropping songs with zero popularity,
     as it basically amounts to cheating. In reality, many songs have zero popularity
     and our model should be able to handle it and ideally be able to predict it, too.
     Therefore, we have made this an option, which is disabled by default.
   * We furthermore added the threshold for a song being considered as popular
     as an explicit parameter. We might want to change it down the line, noting 
     that any hard threshold is somewhat arbitrary in the end. Perhaps we'll consider treating
     this as a regression problem instead and avoid the problem entirely.
   * We also allow to sample the data to speed up development. In many stages
     of development, we don't care about learning a very good model that considers
     as much data as possible. So for experimentation, where we want to check
     only if our code works and perhaps get a rough estimate of how well it works,
     we'll be using a smaller sample of the data.
   We will only be using a single data set in this project, but we will still experiment with 
   different variations thereof. Therefore, we need to make the respective parameters 
   explicit. The parameters collectively determine the concrete manifestation of the data
   and recording them will therefore be akin to data versioning.

   ```python
   class Dataset:
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

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        df = pd.read_csv(config.csv_data_path()).dropna()
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        df[COL_GEN_POPULARITY_CLASS] = df[COL_POPULARITY].apply(lambda x: CLASS_POPULAR if x >= self.threshold_popular else CLASS_UNPOPULAR)
        return df

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        return df.drop(columns=COL_GEN_POPULARITY_CLASS), df[COL_GEN_POPULARITY_CLASS]
   ```

 * **We shall prefer to keep the data
   untouched and make models decide what to do with it**. 
   We don't need to drop or modify any data columns from the get-go. Models can just 
   choose to use only a subset of the data. Different models shall
   be able to do different things entirely.
 * **We avoid the use of constant literals in our code and use named constants instead.**
     * Referring to data columns via string literals is prone to errors; typos happen.
     * With strings, we would get no assistance from our IDE when we're in doubt as to how a column is named. 
       By adding constants to our code with a well-defined identifier scheme (all identifiers start with `COL_`), we
       get optimal assistance from our IDE: By typing `COL_` and asking for an auto-completion,
       we get a list of possible options. 
     * If the column names should ever change, we have a single place in which we would need to 
       update them (though this won't be an issue in this toy project, of course).
     * In addition to the original columns, we introduce constants for the columns we add (such as 
       the class column, with dedicated prefix `COL_GEN`) and introduce semantic groups (with the dedicated prefix `COLS_`) that could
       come in handy later on. 
 * We simplified the generation of the class column using the `apply` function.
 * We have made explicit the set of features that is actually being used by the models (in variable `cols_used_by_models`).
   Previously, it was simply the set of numerical columns that remained after some drops and projections.
   And we can already see that applying `StandardScaler` to columns that are actually categorical (such as the key and the mode),
   is, at the very least, questionable. We shall do something about this in a subsequent step.

# Principles Addressed in this Step

* Find the right abstractions
* Expose parametrisation
* (Version data)