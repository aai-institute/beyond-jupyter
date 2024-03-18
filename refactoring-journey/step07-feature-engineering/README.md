# Step 7: Feature Engineering

If we look at the reality of song popularity, the artist's identity matters more
than anything: 
A song by a popular artist is far more likely to be popular than a song by an unpopular/unknown
artist - even if its attributes are exactly the same otherwise.
While we could theoretically use the artist's identity as a categorical feature,
this is hardly practical owing to the very large number of artists.

In this step, we thus engineer a custom feature that implements the same notion in a
more practical manner:
We add as a feature the relative frequency with which *other* songs by the same
artist (if any) are popular.
`FeatureGeneratorMeanArtistPopularity` implements this feature and is registered
under `FeatureName.MEAN_ARTIST_FREQ_POPULAR` in the feature generator registry (see updated module [features](songpop/features.py)).
The particular semantics necessitate a differentiation between the learning case
and the inference case:
  * For the inference case, we can simply use the relative frequency we observed
    for the artist in the entire training set.
  * For the learning case, we must exclude the current data point (as including it  
    would constitute an obvious data leak).

In either case, it is then possible for the feature to be undefined:
  * During inference, the artist in question may not have appeared in the training set.
  * During training, an artist may have but a single song in the training set.

We use XGBoost's gradient-boosted decision trees as a type of model that explicitly
supports incomplete data. 

```python
    @classmethod
    def create_xgb(cls, name_suffix="", add_features: Sequence[FeatureName] = (), **kwargs):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, *add_features, registry=registry)
        return XGBGradientBoostedVectorClassificationModel(**kwargs) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"XGBoost{name_suffix}")
```

We can add our newly introduced feature via the parameter `add_features` and 
consider two concrete models - one which includes the feature and
one which does not.

```python
    models = [
        ...
        ModelFactory.create_xgb(),
        ModelFactory.create_xgb("-meanArtistFreqPopular", [FeatureName.MEAN_ARTIST_FREQ_POPULAR]),
    ]
```

As far as the other models are concerned, we could apply a feature transformation involving
imputation if we  wanted to support the new feature.


## Principles Addressed in this Step

* Expose parametrisation


<hr>

[Next Step](../step08-high-level-evaluation/README.md)
