# Step 7: Feature Engineering

If we look at the reality of song popularity, the artist's identity matters more
than anything: 
A song by a popular artist is far more likely to be popular than a song by an unpopular/unknown
artist - even if its attributes are exactly the same otherwise.
While we could theoretically use the artist's identity as a categorical feature,
it is hardly practical owing to the very large number of artists.

In this step, we engineer a custom feature that implements the same notion in a
more practical manner:
We add as a feature the relative frequency with which *other* songs by the same
artist are popular.
`FeatureGeneratorMeanArtistPopularity` implements this feature and is registered
under `FeatureName.MEAN_ARTIST_FREQ_POPULAR` in the feature generator registry.
The particular semantics necessitate a differentiation between the learning case
and the inference case:
  * For the inference case, we can simply use the relative frequency we observed
    for the artist in the entire training set.
  * For the training case, we must exclude the current data point (as including it  
    would constitute a data leak).

In either case, it is then possible for the feature to be undefined:
  * During inference, the artist in question may not have appeared in the training set.
  * During training, an artist may have but a single song in the training set.

We use XGBoost's gradient-boosted decision trees to have models that explicitly
support incomplete data, adding two models - one which includes the feature and
one which does not.
For the other models, we could apply a feature transformation that performs
imputation.