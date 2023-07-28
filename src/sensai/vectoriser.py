from enum import Enum
import logging
from typing import Callable, Union, TypeVar, Generic, Sequence, List, Tuple, Iterable, Dict, Hashable, Optional

import numpy as np

from .util import LogTime
from .util.pickle import setstate
from .util.string import list_string, ToStringMixin

T = TypeVar("T")

log = logging.getLogger(__name__)


class Vectoriser(Generic[T], ToStringMixin):
    """
    A vectoriser represents a method for the conversion of instances of some type T into
    vectors, i.e. one-dimensional (numeric) arrays, or (in the special case of a 1D vector) scalars
    """

    log = log.getChild(__qualname__)

    def __init__(self, f: Callable[[T], Union[float, np.ndarray, list]], transformer=None):
        """
        :param f: the function which maps from an instance of T to an array/list/scalar
        :param transformer: an optional transformer (e.g. instance of one of the classes in sklearn.preprocessing)
            which can be used to transform/normalise the generated arrays
        """
        self._fn = f
        self.transformer = transformer
        self._resultType = None
        self.name = None

    def __setstate__(self, state):
        setstate(Vectoriser, self, state, new_optional_properties=["_resultType", "name"], renamed_properties={"f": "_fn"})

    def _tostring_exclude_private(self) -> bool:
        return True

    def set_name(self, name):
        self.name = name

    def get_name(self):
        """
        :return: the name of this feature generator, which may be a default name if the name has not been set. Note that feature generators
            created by a FeatureGeneratorFactory always get the name with which the generator factory was registered.
        """
        if self.name is None:
            return f"{self.__class__.__name__}-{id(self)}"
        return self.name

    def fit(self, items: Iterable[T]):
        if self.transformer is not None:
            values = [self._f(item) for item in items]
            self.transformer.fit(np.array(values))

    def _f(self, x) -> np.array:
        y = self._fn(x)

        if self._resultType is None:
            self._resultType = self.ResultType.from_value(y)

        if self._resultType == self.ResultType.LIST:
            y = np.array(y)
        elif self._resultType == self.ResultType.SCALAR:
            y = np.array([y])

        return y

    def apply(self, item: T, transform=True) -> np.array:
        """
        :param item: the item to be vectorised
        :param transform: whether to apply this instance's transformer (if any)
        :return: a vector
        """
        value = self._f(item)
        if self.transformer is not None and transform:
            value = self.transformer.transform([value])[0]
        return value

    def apply_multi(self, items: Iterable[T], transform=True, use_cache=False, verbose=False) -> List[np.array]:
        """
        Applies this vectoriser to multiple items at once.
        Especially for cases where this vectoriser uses a transformer, this method is significantly faster than
        calling apply repeatedly.

        :param items: the items to be vectorised
        :param transform: whether to apply this instance's transformer (if any)
        :param use_cache: whether to apply caching of the value function f given at construction (keeping track of outputs for
            each input object id), which can significantly speed up computation in cases where an items appears more than
            once in the collection of items
        :param verbose: whether to generate log messages
        :return: a list of vectors
        """
        if verbose:
            self.log.info(f"Applying {self}")

        with LogTime("Application", enabled=verbose, logger=self.log):
            if not use_cache:
                compute_value = self._f
            else:
                cache = {}

                def compute_value(x):
                    key = id(x)
                    value = cache.get(key)
                    if value is None:
                        value = self._f(x)
                        cache[key] = value
                    return value

            values = [compute_value(x) for x in items]
            if self.transformer is not None and transform:
                values = self.transformer.transform(values)
            return values

    class ResultType(Enum):
        SCALAR = 0
        LIST = 1
        NUMPY_ARRAY = 2

        @classmethod
        def from_value(cls, y):
            if type(y) == list:
                return cls.LIST
            elif np.isscalar(y):
                return cls.SCALAR
            elif isinstance(y, np.ndarray):
                return cls.NUMPY_ARRAY
            else:
                raise ValueError(f"Received unhandled value of type {type(y)}")


class EmptyVectoriser(Vectoriser):
    def __init__(self):
        super().__init__(self._create_empty_vector)

    # noinspection PyUnusedLocal
    @staticmethod
    def _create_empty_vector(x):
        return np.zeros(0)


class SequenceVectoriser(Generic[T], ToStringMixin):
    """
    Supports the application of Vectorisers to sequences of objects of some type T, where each object of type T is
    mapped to a vector (1D array) by the vectorisers.
    A SequenceVectoriser is fitted by fitting the underlying Vectorisers. In order to obtain the instances of T that
    are used for training, we take into consideration the fact that the sequences of T may overlap and thus training
    is performed on the set of unique instances.
    """

    log = log.getChild(__qualname__)

    class FittingMode(Enum):
        """
        Determines how the individual vectorisers are fitted based on several sequences of objects of type T that are given.
        If NONE, no fitting is performed, otherwise the mode determines how a single sequence of objects of type T for fitting
        is obtained from the collection of sequences: either by forming the set of unique objects from the sequences (UNIQUE)
        """
        NONE = "none"  # no fitting is performed
        UNIQUE = "unique"  # use collection of unique items
        CONCAT = "concat"  # use collection obtained by concatenating all sequences using numpy.concatenate

    def __init__(self, vectorisers: Union[Sequence[Vectoriser[T]], Vectoriser[T]], fitting_mode: FittingMode = FittingMode.UNIQUE):
        """
        :param vectorisers: zero or more vectorisers that are to be applied. If more than one vectoriser is supplied,
            vectors are generated from input instances of type T by concatenating the results of the vectorisers in
            the order the vectorisers are given.
        """
        self.fittingMode = fitting_mode
        if isinstance(vectorisers, Vectoriser):
            self.vectorisers = [vectorisers]
        else:
            self.vectorisers = vectorisers
        if len(self.vectorisers) == 0:
            self.vectorisers = [EmptyVectoriser()]

    def __setstate__(self, state):
        state["fittingMode"] = state.get("fittingMode", self.FittingMode.UNIQUE)
        setstate(SequenceVectoriser, self, state)

    def fit(self, data: Iterable[Sequence[T]]):
        log.debug(f"Fitting {self}")
        if self.fittingMode == self.FittingMode.NONE:
            return
        if self.fittingMode == self.FittingMode.UNIQUE:
            items = set()
            for seq in data:
                items.update(seq)
        elif self.fittingMode == self.FittingMode.CONCAT:
            items = np.concatenate(data)
        else:
            raise ValueError(self.fittingMode)
        for v in self.vectorisers:
            log.debug(f"Fitting {v}")
            v.fit(items)

    def apply(self, seq: Sequence[T], transform=True) -> List[np.array]:
        """
        Applies vectorisation to the given sequence of objects

        :param seq: the sequence to vectorise
        :param transform: whether to apply any post-vectorisation transformers
        :return:
        """
        vectors_list = []
        for item in seq:
            vectors = [vec.apply(item, transform=transform) for vec in self.vectorisers]
            conc = np.concatenate(vectors, axis=0)
            vectors_list.append(conc)
        return vectors_list

    def apply_multi(self, sequences: Iterable[Sequence[T]], use_cache=False, verbose=False) -> Tuple[List[List[np.array]], List[int]]:
        """
        Applies this vectoriser to multiple sequences of objects of type T, where each sequence is mapped to a sequence
        of 1D arrays.
        This method can be significantly faster than multiple applications of apply, especially in cases where the vectorisers
        use transformers.

        :param sequences: the sequences to vectorise
        :param use_cache: whether to apply caching of the value functions of contained vectorisers (keeping track of outputs for
            each input object id), which can significantly speed up computation in cases where the given sequences contain individual
            items more than once
        :param verbose: whether to generate log messages
        :return: a pair (vl, l) where vl is a list of lists of vectors/arrays and l is a list of integers containing the lengths
            of the sequences
        """
        if verbose:
            self.log.info(f"Applying {self} (useCache={use_cache})")

        lengths = [len(s) for s in sequences]

        if verbose:
            self.log.info("Generating combined sequence")
        combined_seq = []
        for seq in sequences:
            combined_seq.extend(seq)

        individual_vectoriser_results = [vectoriser.apply_multi(combined_seq, use_cache=use_cache, verbose=verbose)
            for vectoriser in self.vectorisers]
        conc_vectors = [np.concatenate(x, axis=0) for x in zip(*individual_vectoriser_results)]

        vector_sequences = []
        idx_start = 0
        for l in lengths:
            vector_sequences.append(conc_vectors[idx_start:idx_start+l])
            idx_start += l

        return vector_sequences, lengths

    def apply_multi_with_padding(self, sequences: Sequence[Sequence[T]], use_cache=False, verbose=False) \
            -> Tuple[List[List[np.array]], List[int]]:
        """
        Applies this vectoriser to multiple sequences of objects of type T, where each sequence is mapped to a sequence
        of 1D arrays.
        Sequences are allowed to vary in length. for shorter sequences, 0-vectors are appended until the maximum sequence length
        is reached (padding).

        :param sequences: the sequences to vectorise
        :param use_cache: whether to apply caching of the value functions of contained vectorisers (keeping track of outputs for
            each input object id), which can significantly speed up computation in cases where the given sequences contain individual
            items more than once
        :param verbose: whether to generate log messages
        :return: a pair (vl, l) where vl is a list of lists of vectors/arrays, each list having the same length, and l is a list of
            integers containing the original unpadded lengths of the sequences
        """
        result, lengths = self.apply_multi(sequences, use_cache=use_cache, verbose=verbose)
        if verbose:
            self.log.info("Applying padding")
        max_length = max(lengths)
        dim = len(result[0][0])
        dummy_vec = np.zeros((dim,))
        for seq in result:
            for i in range(max_length - len(seq)):
                seq.append(dummy_vec)
        return result, lengths

    def get_vector_dim(self, seq: Sequence[T]):
        """
        Determines the dimensionality of generated vectors by applying the vectoriser to the given sequence

        :param seq: the sequence
        :return: the number of dimensions in generated output vectors (per item)
        """
        return len(self.apply(seq, transform=False)[0])


class VectoriserRegistry:
    def __init__(self):
        self._factories: Dict[Hashable, Callable[[Callable], Vectoriser]] = {}

    def get_available_vectorisers(self):
        return list(self._factories.keys())

    @staticmethod
    def _name(name: Hashable):
        # for enums, which have .name, use the name only, because it is less problematic to persist
        if hasattr(name, "name"):
            name = name.name
        return name

    def register_factory(self, name: Hashable, factory: Callable[[Callable], Vectoriser],
            additional_names: Optional[Iterable[Hashable]] = None):
        """
        Registers a vectoriser factory which can subsequently be referenced via their name

        :param name: the name (which can, in particular, be a string or an enum item)
        :param factory: the factory, which takes the default transformer factory as an argument
        :param additional_names: (optional) additional names under which to register the factory
        """
        self._register_factory(name, factory)
        if additional_names is not None:
            for n in additional_names:
                self._register_factory(n, factory)

    def _register_factory(self, name: Hashable, factory):
        name = self._name(name)
        if name in self._factories:
            raise ValueError(f"Vectoriser factory for name '{name}' already registered")
        self._factories[name] = factory

    def get_vectoriser(self, name: Hashable, default_transformer_factory: Callable) -> Vectoriser:
        """
        Creates a vectoriser from a name, which must have been previously registered.

        :param name: the name (which can, in particular, be a string or an enum item)
        :param default_transformer_factory: the default transformer factory
        :return: a new vectoriser instance
        """
        name = self._name(name)
        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"No vectoriser factory registered for name '{name}': known names: {list_string(self._factories.keys())}. "
                             f"Register the factory first.")
        instance = factory(default_transformer_factory)
        instance.set_name(name)
        return instance

    def get_vectorisers(self, names: List[Hashable], default_transformer_factory: Callable) -> List[Vectoriser]:
        return [self.get_vectoriser(name, default_transformer_factory) for name in names]
