import logging
from typing import Callable, Dict, TYPE_CHECKING, Hashable, Union

import pandas as pd

from . import FeatureGenerator, MultiFeatureGenerator
from ..data_transformation import DFTNormalisation, DFTOneHotEncoder
from ..util.string import list_string

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class FeatureGeneratorRegistry:
    """
    Represents a registry for (named) feature generator factories
    """
    def __init__(self, use_singletons=False):
        """
        :param use_singletons: if True, internally maintain feature generator singletons, such that there is at most one
            instance for each name/key
        """
        self._feature_generator_factories: Dict[Hashable, Callable[[], FeatureGenerator]] = {}
        self._feature_generator_singletons: Dict[Hashable, FeatureGenerator] = {}
        self._use_singletons = use_singletons

    @property
    def available_features(self):
        return list(self._feature_generator_factories.keys())

    @staticmethod
    def _name(name: Hashable):
        # for enums, which have .name, use the name only, because it is less problematic to persist
        if hasattr(name, "name"):
            name = name.name
        return name

    def register_factory(self, name: Hashable, factory: Callable[[], FeatureGenerator]):
        """
        Registers a feature generator factory which can subsequently be referenced by models via their name/hashable key

        :param name: the name/key (which can, in particular, be a string or an Enum item). Especially for larger projects
            the use of an Enum is recommended (for optimal IDE support)
        :param factory: the factory
        """
        name = self._name(name)
        if name in self._feature_generator_factories:
            raise ValueError(f"Generator for name '{name}' already registered")
        self._feature_generator_factories[name] = factory

    def get_feature_generator(self, name) -> FeatureGenerator:
        """
        Creates a feature generator from a name, which must have been previously registered.
        The name of the returned feature generator (as returned by getName()) is set to name.

        :param name: the name (which can, in particular, be a string or an enum item)
        :return: a new feature generator instance (or existing instance for the case where useSingletons is enabled)
        """
        name = self._name(name)
        generator = self._feature_generator_singletons.get(name)
        if generator is None:
            factory = self._feature_generator_factories.get(name)
            if factory is None:
                raise ValueError(f"No factory registered for name '{name}': known names: {list_string(self._feature_generator_factories.keys())}. Use registerFeatureGeneratorFactory to register a new feature generator factory.")
            generator = factory()
            generator.set_name(name)
            if self._use_singletons:
                self._feature_generator_singletons[name] = generator
        return generator

    def collect_features(self, *feature_generators_or_names: Union[Hashable, FeatureGenerator]) -> "FeatureCollector":
        """
        Creates a feature collector for the given feature names/keys/instances, which can subsequently be added to a model.

        :param feature_generators_or_names: feature names/keys known to this registry or feature generator instances
        """
        return FeatureCollector(*feature_generators_or_names, registry=self)


class FeatureCollector(object):
    """
    A feature collector which facilitates the collection of features that shall be used by a model as well as the
    generation of commonly used feature transformers that are informed by the features' meta-data.
    """

    def __init__(self,
            *feature_generators_or_names: Union[Hashable, FeatureGenerator],
            registry: FeatureGeneratorRegistry = None):
        """
        :param feature_generators_or_names: generator names/keys (known to the registry) or generator instances
        :param registry: the feature generator registry for the case where names/keys are passed
        """
        self._feature_generators_or_names = feature_generators_or_names
        self._registry = registry
        self._multi_feature_generator = self._create_multi_feature_generator()

    def get_multi_feature_generator(self) -> MultiFeatureGenerator:
        return self._multi_feature_generator

    def get_normalisation_rules(self, include_generated_categorical_rules=True):
        return self.get_multi_feature_generator().get_normalisation_rules(
            include_generated_categorical_rules=include_generated_categorical_rules)

    def get_categorical_feature_name_regex(self) -> str:
        """
        :return: a regular expression that matches all known categorical feature names
        """
        return self.get_multi_feature_generator().get_categorical_feature_name_regex()

    def _create_multi_feature_generator(self):
        feature_generators = []
        for f in self._feature_generators_or_names:
            if isinstance(f, FeatureGenerator):
                feature_generators.append(f)
            else:
                if self._registry is None:
                    raise Exception(f"Received feature name '{f}' instead of instance but no registry to perform the lookup")
                feature_generators.append(self._registry.get_feature_generator(f))
        return MultiFeatureGenerator(*feature_generators)

    def create_dft_normalisation(self, default_transformer_factory=None, require_all_handled=True, inplace=False) -> DFTNormalisation:
        """
        Creates a feature transformer that will apply normalisation to all supported (numeric) features

        :param default_transformer_factory: a factory for the creation of transformer instances (which implements the
            API used by sklearn.preprocessing, e.g. StandardScaler) that shall be used to create a transformer for all
            rules that do not specify a particular transformer.
            The default transformer will only be applied to columns matched by such rules, unmatched columns will
            not be transformed.
            Use SkLearnTransformerFactoryFactory to conveniently create a factory.
        :param require_all_handled: whether to raise an exception if not all columns are matched by a rule
        :param inplace: whether to apply data frame transformations in-place
        :return: the transformer
        """
        return DFTNormalisation(self.get_normalisation_rules(), default_transformer_factory=default_transformer_factory,
            require_all_handled=require_all_handled, inplace=inplace)

    def create_dft_one_hot_encoder(self, ignore_unknown=False, inplace=False):
        """
        Creates a feature transformer that will apply one-hot encoding to all the features that are known to be categorical

        :param inplace: whether to perform the transformation in-place
        :param ignore_unknown: if True and an unknown category is encountered during transform, the resulting one-hot
            encoded columns for this feature will be all zeros. if False, an unknown category will raise an error.
        :return: the transformer
        """
        return DFTOneHotEncoder(self.get_categorical_feature_name_regex(), ignore_unknown=ignore_unknown, inplace=inplace)

    def create_feature_transformer_normalisation(self, default_transformer_factory=None, require_all_handled=True, inplace=False) \
            -> DFTNormalisation:
        """
        Creates a feature transformer that will apply normalisation to all supported (numeric) features.
        Alias of create_dft_normalisation.

        :param default_transformer_factory: a factory for the creation of transformer instances (which implements the
            API used by sklearn.preprocessing, e.g. StandardScaler) that shall be used to create a transformer for all
            rules that do not specify a particular transformer.
            The default transformer will only be applied to columns matched by such rules, unmatched columns will
            not be transformed.
            Use SkLearnTransformerFactoryFactory to conveniently create a factory.
        :param require_all_handled: whether to raise an exception if not all columns are matched by a rule
        :param inplace: whether to apply data frame transformations in-place
        :return: the transformer
        """
        return self.create_dft_normalisation(default_transformer_factory=default_transformer_factory,
            require_all_handled=require_all_handled, inplace=inplace)

    def create_feature_transformer_one_hot_encoder(self, ignore_unknown=False, inplace=False):
        """
        Creates a feature transformer that will apply one-hot encoding to all the features that are known to be categorical.
        Alias of create_dft_one_hot_encoder.

        :param inplace: whether to perform the transformation in-place
        :param ignore_unknown: if True and an unknown category is encountered during transform, the resulting one-hot
            encoded columns for this feature will be all zeros. if False, an unknown category will raise an error.
        :return: the transformer
        """
        return self.create_dft_one_hot_encoder(ignore_unknown=ignore_unknown, inplace=inplace)
