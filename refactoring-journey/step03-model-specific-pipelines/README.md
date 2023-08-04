In this step, we make another critical step towards more flexibility.
We shall strongly associate the data processing pipeline with the models in order
to enable different models to use entirely different pipelines.
Importantly, different models might
  * use a different set of features
  * use different representations of these features.

So far, all models use exactly the same features and use
the same StandardScaler-induced representations of these features.
Clearly, this is a compromise, as some of the models could, potentially,
make good use of categorical features such as genre of the song; 
and we have already pointed out that the use of StandardScaler is not 
necessarily optimal for all the features it is currently being applied to.
By making the input pipeline a part of the model, we gain the flexibility of
trying out new models that don't stick to the current limitations down the line.

We introduce a model factory that is able to create instances of the four different models
we currently consider. 
We have named the factory functions in a way that indicates that they are the original
models (with suffix `_orig`), preparing for the future case where we will have
several additional models.

