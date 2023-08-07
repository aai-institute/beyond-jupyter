import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Sequence, Union, Dict, Callable, Any, Optional, Set

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder

from .sklearn_transformer import SkLearnTransformerProtocol
from ..columngen import ColumnGenerator
from ..util import flatten_arguments, count_not_none
from ..util.pandas import DataFrameColumnChangeTracker
from ..util.pickle import setstate
from ..util.string import or_regex_group, ToStringMixin

from typing import TYPE_CHECKING

from ..util.version import Version

if TYPE_CHECKING:
    from ..featuregen import FeatureGenerator

log = logging.getLogger(__name__)


class DataFrameTransformer(ABC, ToStringMixin):
    """
    Base class for data frame transformers, i.e. objects which can transform one data frame into another
    (possibly applying the transformation to the original data frame - in-place transformation).
    A data frame transformer may require being fitted using training data.
    """
    def __init__(self):
        self._name = f"{self.__class__.__name__}-{id(self)}"
        self._isFitted = False
        self._columnChangeTracker: Optional[DataFrameColumnChangeTracker] = None
        self._paramInfo = {}  # arguments passed to init that are not saved otherwise can be persisted here

    # for backwards compatibility with persisted DFTs based on code prior to commit 7088cbbe
    # They lack the __isFitted attribute and we assume that each such DFT was fitted
    def __setstate__(self, d):
        d["_name"] = d.get("_name", f"{self.__class__.__name__}-{id(self)}")
        d["_isFitted"] = d.get("_isFitted", True)
        d["_columnChangeTracker"] = d.get("_columnChangeTracker", None)
        d["_paramInfo"] = d.get("_paramInfo", {})
        self.__dict__ = d

    def _tostring_exclude_private(self) -> bool:
        return True

    def get_name(self) -> str:
        """
        :return: the name of this dft transformer, which may be a default name if the name has not been set.
        """
        return self._name

    def set_name(self, name: str):
        self._name = name

    def with_name(self, name: str):
        self.set_name(name)
        return self

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._columnChangeTracker = DataFrameColumnChangeTracker(df)
        if not self.is_fitted():
            raise Exception(f"Cannot apply a DataFrameTransformer which is not fitted: "
                            f"the df transformer {self.get_name()} requires fitting")
        df = self._apply(df)
        self._columnChangeTracker.track_change(df)
        return df

    def info(self):
        return {
            "name": self.get_name(),
            "changeInColumnNames": self._columnChangeTracker.column_change_string() if self._columnChangeTracker is not None else None,
            "isFitted": self.is_fitted(),
        }

    def fit(self, df: pd.DataFrame):
        self._fit(df)
        self._isFitted = True

    def is_fitted(self):
        return self._isFitted

    def fit_apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.apply(df)

    def to_feature_generator(self, categorical_feature_names: Optional[Union[Sequence[str], str]] = None,
            normalisation_rules: Sequence['DFTNormalisation.Rule'] = (),
            normalisation_rule_template: 'DFTNormalisation.RuleTemplate' = None,
            add_categorical_default_rules=True):
        # need to import here to prevent circular imports
        from ..featuregen import FeatureGeneratorFromDFT
        return FeatureGeneratorFromDFT(
            self, categorical_feature_names=categorical_feature_names, normalisation_rules=normalisation_rules,
            normalisation_rule_template=normalisation_rule_template, add_categorical_default_rules=add_categorical_default_rules
        )


class DFTFromFeatureGenerator(DataFrameTransformer):
    def _fit(self, df: pd.DataFrame):
        self.fgen.fit(df, ctx=None)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fgen.generate(df)

    def __init__(self, fgen: "FeatureGenerator"):
        super().__init__()
        self.fgen = fgen
        self.set_name(f"{self.__class__.__name__}[{self.fgen.get_name()}]")


class InvertibleDataFrameTransformer(DataFrameTransformer, ABC):
    @abstractmethod
    def apply_inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_inverse(self) -> "InverseDataFrameTransformer":
        """
        :return: a transformer whose (forward) transformation is the inverse transformation of this DFT
        """
        return InverseDataFrameTransformer(self)


class RuleBasedDataFrameTransformer(DataFrameTransformer, ABC):
    """Base class for transformers whose logic is entirely based on rules and does not need to be fitted to data"""

    def _fit(self, df: pd.DataFrame):
        pass

    def fit(self, df: pd.DataFrame):
        pass

    def is_fitted(self):
        return True


class InverseDataFrameTransformer(RuleBasedDataFrameTransformer):
    def __init__(self, invertible_dft: InvertibleDataFrameTransformer):
        super().__init__()
        self.invertibleDFT = invertible_dft

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.invertibleDFT.apply_inverse(df)


class DataFrameTransformerChain(DataFrameTransformer):
    """
    Supports the application of a chain of data frame transformers.
    During fit and apply each transformer in the chain receives the transformed output of its predecessor.
    """

    def __init__(self, *data_frame_transformers: Union[DataFrameTransformer, List[DataFrameTransformer]]):
        super().__init__()
        self.dataFrameTransformers = flatten_arguments(data_frame_transformers)

    def __len__(self):
        return len(self.dataFrameTransformers)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in self.dataFrameTransformers:
            df = transformer.apply(df)
        return df

    def _fit(self, df: pd.DataFrame):
        if len(self.dataFrameTransformers) == 0:
            return
        for transformer in self.dataFrameTransformers[:-1]:
            df = transformer.fit_apply(df)
        self.dataFrameTransformers[-1].fit(df)

    def is_fitted(self):
        return all([dft.is_fitted() for dft in self.dataFrameTransformers])

    def get_names(self) -> List[str]:
        """
        :return: the list of names of all contained feature generators
        """
        return [transf.get_name() for transf in self.dataFrameTransformers]

    def info(self):
        info = super().info()
        info["chainedDFTTransformerNames"] = self.get_names()
        info["length"] = len(self)
        return info

    def find_first_transformer_by_type(self, cls) -> Optional[DataFrameTransformer]:
        for dft in self.dataFrameTransformers:
            if isinstance(dft, cls):
                return dft
        return None

    def append(self, t: DataFrameTransformer):
        self.dataFrameTransformers.append(t)


class DFTRenameColumns(RuleBasedDataFrameTransformer):
    def __init__(self, columns_map: Dict[str, str]):
        """
        :param columns_map: dictionary mapping old column names to new names
        """
        super().__init__()
        self.columnsMap = columns_map

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.columnsMap)


class DFTConditionalRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a boolean function to one of the columns and retaining only the rows
    for which the function returns True
    """
    def __init__(self, column: str, condition: Callable[[Any], bool]):
        super().__init__()
        self.column = column
        self.condition = condition

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.column].apply(self.condition)]


class DFTInSetComparisonRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame on the selected column and retains only the rows for which the value is in the setToKeep
    """
    def __init__(self, column: str, set_to_keep: Set):
        super().__init__()
        self.setToKeep = set_to_keep
        self.column = column

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.column].isin(self.setToKeep)]

    def info(self):
        info = super().info()
        info["column"] = self.column
        info["setToKeep"] = self.setToKeep
        return info


class DFTNotInSetComparisonRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame on the selected column and retains only the rows for which the value is not in the setToDrop
    """
    def __init__(self, column: str, set_to_drop: Set):
        super().__init__()
        self.setToDrop = set_to_drop
        self.column = column

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[~df[self.column].isin(self.setToDrop)]

    def info(self):
        info = super().info()
        info["column"] = self.column
        info["setToDrop"] = self.setToDrop
        return info


class DFTVectorizedConditionalRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a vectorized condition on the selected column and retaining only the rows
    for which it returns True
    """
    def __init__(self, column: str, vectorized_condition: Callable[[pd.Series], Sequence[bool]]):
        super().__init__()
        self.column = column
        self.vectorizedCondition = vectorized_condition

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.vectorizedCondition(df[self.column])]

    def info(self):
        info = super().info()
        info["column"] = self.column
        return info


class DFTRowFilter(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a condition function to each row and retaining only the rows
    for which it returns True
    """
    def __init__(self, condition: Callable[[Any], bool]):
        super().__init__()
        self.condition = condition

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.apply(self.condition, axis=1)]


class DFTModifyColumn(RuleBasedDataFrameTransformer):
    """
    Modifies a column specified by 'column' using 'columnTransform'
    """
    def __init__(self, column: str, column_transform: Union[Callable, np.ufunc]):
        """
        :param column: the name of the column to be modified
        :param column_transform: a function operating on single cells or a Numpy ufunc that applies to an entire Series
        """
        super().__init__()
        self.column = column
        self.columnTransform = column_transform

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = df[self.column].apply(self.columnTransform)
        return df


class DFTModifyColumnVectorized(RuleBasedDataFrameTransformer):
    """
    Modifies a column specified by 'column' using 'columnTransform'. This transformer can be used to utilise Numpy vectorisation for
    performance optimisation.
    """
    def __init__(self, column: str, column_transform: Callable[[np.ndarray], Union[Sequence, pd.Series, np.ndarray]]):
        """
        :param column: the name of the column to be modified
        :param column_transform: a function that takes a Numpy array and from which the returned value will be assigned to the column as
            a whole
        """
        super().__init__()
        self.column = column
        self.columnTransform = column_transform

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = self.columnTransform(df[self.column].values)
        return df


class DFTOneHotEncoder(DataFrameTransformer):
    def __init__(self, columns: Optional[Union[str, Sequence[str]]],
            categories: Union[List[np.ndarray], Dict[str, np.ndarray]] = None, inplace=False, ignore_unknown=False,
            array_valued_result=False):
        """
        One hot encode categorical variables

        :param columns: list of names or regex matching names of columns that are to be replaced by a list one-hot encoded columns each
            (or an array-valued column for the case where useArrayValues=True);
            If None, then no columns are actually to be one-hot-encoded
        :param categories: numpy arrays containing the possible values of each of the specified columns (for case where sequence is
            specified in 'columns') or dictionary mapping column name to array of possible categories for the column name.
            If None, the possible values will be inferred from the columns
        :param inplace: whether to perform the transformation in-place
        :param ignore_unknown: if True and an unknown category is encountered during transform, the resulting one-hot
            encoded columns for this feature will be all zeros. if False, an unknown category will raise an error.
        :param array_valued_result: whether to replace the input columns by columns of the same name containing arrays as values
            instead of creating a separate column per original value
        """
        super().__init__()
        self._paramInfo["columns"] = columns
        self._paramInfo["inferCategories"] = categories is None
        self.oneHotEncoders = None
        if columns is None:
            self._columnsToEncode = []
            self._columnNameRegex = "$"
        elif type(columns) == str:
            self._columnNameRegex = columns
            self._columnsToEncode = None
        else:
            self._columnNameRegex = or_regex_group(columns)
            self._columnsToEncode = columns
        self.inplace = inplace
        self.arrayValuedResult = array_valued_result
        self.handleUnknown = "ignore" if ignore_unknown else "error"
        if categories is not None:
            if type(categories) == dict:
                self.oneHotEncoders = {col: OneHotEncoder(categories=[np.sort(categories)], handle_unknown=self.handleUnknown,
                    **self._sparse_kwargs()) for col, categories in categories.items()}
            else:
                if len(columns) != len(categories):
                    raise ValueError(f"Given categories must have the same length as columns to process")
                self.oneHotEncoders = {col: OneHotEncoder(categories=[np.sort(categories)], handle_unknown=self.handleUnknown,
                    **self._sparse_kwargs()) for col, categories in zip(columns, categories)}

    @staticmethod
    def _sparse_kwargs(sparse=False):
        if Version(sklearn).is_at_least(1, 2):
            return dict(sparse_output=sparse)
        else:
            return dict(sparse=sparse)

    def __setstate__(self, state):
        if "arrayValuedResult" not in state:
            state["arrayValuedResult"] = False
        super().__setstate__(state)

    def _tostring_additional_entries(self) -> Dict[str, Any]:
        d = super()._tostring_additional_entries()
        d["columns"] = self._paramInfo.get("columns")
        return d

    def _fit(self, df: pd.DataFrame):
        if self._columnsToEncode is None:
            self._columnsToEncode = [c for c in df.columns if re.fullmatch(self._columnNameRegex, c) is not None]
            if len(self._columnsToEncode) == 0:
                log.warning(f"{self} does not apply to any columns, transformer has no effect; regex='{self._columnNameRegex}'")
        if self.oneHotEncoders is None:
            self.oneHotEncoders = {column: OneHotEncoder(categories=[np.sort(df[column].unique())], handle_unknown=self.handleUnknown,
                **self._sparse_kwargs()) for column in self._columnsToEncode}
        for columnName in self._columnsToEncode:
            self.oneHotEncoders[columnName].fit(df[[columnName]])

    def _apply(self, df: pd.DataFrame):
        if len(self._columnsToEncode) == 0:
            return df

        if not self.inplace:
            df = df.copy()
        for columnName in self._columnsToEncode:
            encoded_array = self.oneHotEncoders[columnName].transform(df[[columnName]])
            if not self.arrayValuedResult:
                df = df.drop(columns=columnName)
                for i in range(encoded_array.shape[1]):
                    df["%s_%d" % (columnName, i)] = encoded_array[:, i]
            else:
                df[columnName] = list(encoded_array)
        return df

    def info(self):
        info = super().info()
        info["inplace"] = self.inplace
        info["handleUnknown"] = self.handleUnknown
        info["arrayValuedResult"] = self.arrayValuedResult
        info.update(self._paramInfo)
        return info


class DFTColumnFilter(RuleBasedDataFrameTransformer):
    """
    A DataFrame transformer that filters columns by retaining or dropping specified columns
    """
    def __init__(self, keep: Union[str, Sequence[str]] = None, drop: Union[str, Sequence[str]] = None):
        super().__init__()
        self.keep = [keep] if type(keep) == str else keep
        self.drop = drop

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.keep is not None:
            df = df[self.keep]
        if self.drop is not None:
            df = df.drop(columns=self.drop)
        return df

    def info(self):
        info = super().info()
        info["keep"] = self.keep
        info["drop"] = self.drop
        return info


class DFTKeepColumns(DFTColumnFilter):
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.keep]


class DFTDRowFilterOnIndex(RuleBasedDataFrameTransformer):
    def __init__(self, keep: Set = None, drop: Set = None):
        super().__init__()
        self.drop = drop
        self.keep = keep

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.keep is not None:
            df = df.loc[self.keep]
        if self.drop is not None:
            df = df.drop(self.drop)
        return df


class DFTNormalisation(DataFrameTransformer):
    """
    Applies normalisation/scaling to a data frame by applying a set of transformation rules, where each
    rule defines a set of columns to which it applies (learning a single transformer based on the values
    of all applicable columns).
    DFTNormalisation ignores N/A values during fitting and application.
    """

    class RuleTemplate:
        def __init__(self,
                skip=False,
                unsupported=False,
                transformer: Optional[SkLearnTransformerProtocol] = None,
                transformer_factory: Callable[[], SkLearnTransformerProtocol] = None,
                independent_columns: Optional[bool] = None):
            """
            Creates a rule template which applies to one or more features/columns (depending on context).
            Use parameters as follows:

                * If the relevant features are already normalised, pass ``skip=True``
                * If the relevant features cannot be normalised (e.g. because they are categorical), pass ``unsupported=True``
                * If the relevant features shall be normalised, the other parameters apply.
                  No parameters, i.e. ``RuleTemplate()``, are an option if ...

                    * a default transformer factory is specified in the :class:`DFTNormalisation` instance and its application
                      is suitable for the relevant set of features.
                      Otherwise, specify either ``transformerFactory`` or ``transformer``.
                    * all relevant features are to be normalised in the same way.
                      Otherwise, specify ``independentColumns=True``.

            :param skip: flag indicating whether no transformation shall be performed on all of the columns (because they are already
                normalised)
            :param unsupported: flag indicating whether normalisation of all columns is unsupported (shall trigger an exception if
                attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to the matching column(s)
                for the case where a transformation is necessary (skip=False, unsupported=False). If None is given, either
                transformerFactory or the containing instance's default factory will be used.
                NOTE: Use an instance only if you want, in particular, the instance to be shared across several models that use the same
                feature with associated rule/rule template (disabling `fit` where appropriate). Otherwise, use a factory.
            :param transformer_factory: a factory for the generation of the transformer instance, which will only be applied if
                `transformer` is not given; if neither `transformer` nor `transformerInstance` are given, the containing instance's default
                factory will be used. See :class:`SkLearnTransformerFactoryFactory` for convenient construction options.
            :param independent_columns: whether, for the case where the rule matches multiple columns, the columns are independent and a
                separate transformation is to be learned for each of them (rather than using the same transformation for all columns and
                learning the transformation from the data of all columns); must be specified for rules matching more than one column,
                None is acceptable only for a single column
            """
            if (skip or unsupported) and count_not_none(transformer, transformer_factory) > 0:
                raise ValueError("Passed transformer or transformerFactory while skip=True or unsupported=True")
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer
            self.transformerFactory = transformer_factory
            self.independentColumns = independent_columns

        def to_rule(self, regex: Optional[str]):
            """
            Convert the template to a rule for all columns matching the regex

            :param regex: a regular expression defining the column the rule applies to
            :return: the resulting Rule
            """
            return DFTNormalisation.Rule(regex, skip=self.skip, unsupported=self.unsupported, transformer=self.transformer,
                transformer_factory=self.transformerFactory, independent_columns=self.independentColumns)

        def to_placeholder_rule(self):
            return self.to_rule(None)

    class Rule(ToStringMixin):
        def __init__(self,
                regex: Optional[str],
                skip=False, unsupported=False,
                transformer: SkLearnTransformerProtocol = None,
                transformer_factory: Callable[[], SkLearnTransformerProtocol] = None,
                array_valued=False,
                fit=True,
                independent_columns: Optional[bool] = None):
            """
            :param regex: a regular expression defining the column(s) the rule applies to.
                If it applies to multiple columns, these columns will be normalised in the same way (using the same normalisation
                process for each column) unless independentColumns=True.
                If None, the rule is a placeholder rule and the regex must be set later via setRegex or the rule will not be applicable.
            :param skip: flag indicating whether no transformation shall be performed on the matching column(s)
            :param unsupported: flag indicating whether normalisation of the matching column(s) is unsupported (shall trigger an exception
                if attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to the matching column(s)
                for the case where a transformation is necessary (skip=False, unsupported=False). If None is given, either
                transformerFactory or the containing instance's default factory will be used.
                NOTE: Use an instance only if you want, in particular, the instance to be shared across several models that use the same
                feature with associated rule/rule template (disabling `fit` where appropriate). Otherwise, use a factory.
            :param transformer_factory: a factory for the generation of the transformer instance, which will only be applied if
                `transformer` is not given; if neither `transformer` nor `transformerInstance` are given, the containing instance's default
                factory will be used. See :class:`SkLearnTransformerFactoryFactory` for convenient construction options.
            :param array_valued: whether the column values are not scalars but arrays (of arbitrary lengths).
                It is assumed that all entries in such arrays are to be normalised in the same way.
                If arrayValued is True, only a single matching column is supported, i.e. the regex must match at most one column.
            :param fit: whether the rule's transformer shall be fitted
            :param independent_columns: whether, for the case where the rule matches multiple columns, the columns are independent and a
                separate transformation is to be learned for each of them (rather than using the same transformation for all columns and
                learning the transformation from the data of all columns); must be specified for rules matching more than one column, None
                is acceptable only for single-column rules
            """
            if skip and (transformer is not None or transformer_factory is not None):
                raise ValueError("skip==True while transformer/transformerFactory is not None")
            self.regex = re.compile(regex) if regex is not None else None
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer
            self.transformerFactory = transformer_factory
            self.arrayValued = array_valued
            self.fit = fit
            self.independentColumns = independent_columns

        def __setstate__(self, state):
            setstate(DFTNormalisation.Rule, self, state, new_default_properties=dict(arrayValued=False, fit=True, independentColumns=False,
                    transformerFactory=None))

        def _tostring_excludes(self) -> List[str]:
            return super()._tostring_excludes() + ["regex"]

        def _tostring_additional_entries(self) -> Dict[str, Any]:
            d = super()._tostring_additional_entries()
            if self.regex is not None:
                d["regex"] = f"'{self.regex.pattern}'"
            return d

        def set_regex(self, regex: str):
            try:
                self.regex = re.compile(regex)
            except Exception as e:
                raise Exception(f"Could not compile regex '{regex}': {e}")

        def matches(self, column: str):
            if self.regex is None:
                raise Exception("Attempted to apply a placeholder rule. Perhaps the feature generator from which the rule originated was "
                                "never applied in order to have the rule instantiated.")
            return self.regex.fullmatch(column) is not None

        def matching_columns(self, columns: Sequence[str]) -> List[str]:
            return [col for col in columns if self.matches(col)]

    def __init__(self, rules: Sequence[Rule], default_transformer_factory=None, require_all_handled=True, inplace=False):
        """
        :param rules: the set of rules; rules are always fitted and applied in the given order.
            A convenient way to obtain a set of rules in the :class:`sensai.vector_model.VectorModel` context is from a
            :class:`sensai.featuregen.FeatureCollector` or :class:`sensai.featuregen.MultiFeatureGenerator`.
        :param default_transformer_factory: a factory for the creation of transformer instances (which implements the
            API used by sklearn.preprocessing, e.g. StandardScaler) that shall be used to create a transformer for all
            rules that do not specify a particular transformer.
            The default transformer will only be applied to columns matched by such rules, unmatched columns will
            not be transformed.
            Use SkLearnTransformerFactoryFactory to conveniently create a factory.
        :param require_all_handled: whether to raise an exception if not all columns are matched by a rule
        :param inplace: whether to apply data frame transformations in-place
        """
        super().__init__()
        self.requireAllHandled = require_all_handled
        self.inplace = inplace
        self._userRules = rules
        self._defaultTransformerFactory = default_transformer_factory
        self._rules = None

    def _tostring_additional_entries(self) -> Dict[str, Any]:
        d = super()._tostring_additional_entries()
        if self._rules is not None:
            d["rules"] = self._rules
        else:
            d["userRules"] = self._userRules
        return d

    def _fit(self, df: pd.DataFrame):
        matched_rules_by_column = {}
        self._rules = []
        for rule in self._userRules:
            matching_columns = rule.matching_columns(df.columns)
            for c in matching_columns:
                if c in matched_rules_by_column:
                    raise Exception(f"More than one rule applies to column '{c}': {matched_rules_by_column[c]}, {rule}")
                matched_rules_by_column[c] = rule

            if len(matching_columns) > 0:
                if rule.unsupported:
                    raise Exception(f"Normalisation of columns {matching_columns} is unsupported according to {rule}. "
                                    f"If you want to make use of these columns, transform them into a supported column before applying "
                                    f"{self.__class__.__name__}.")
                if not rule.skip:
                    if rule.transformer is None:
                        if rule.transformerFactory is not None:
                            rule.transformer = rule.transformerFactory()
                        else:
                            if self._defaultTransformerFactory is None:
                                raise Exception(f"No transformer to fit: {rule} defines no transformer and instance has no transformer "
                                                f"factory")
                            rule.transformer = self._defaultTransformerFactory()
                    if rule.fit:
                        # fit transformer
                        applicable_df = df[sorted(matching_columns)]
                        if rule.arrayValued:
                            if len(matching_columns) > 1:
                                raise Exception(f"Array-valued case is only supported for a single column, "
                                                f"matched {matching_columns} for {rule}")
                            values = np.concatenate(applicable_df.values.flatten())
                            values = values.reshape((len(values), 1))
                        elif rule.independentColumns:
                            values = applicable_df.values
                        else:
                            values = applicable_df.values.flatten()
                            values = values.reshape((len(values), 1))
                        rule.transformer.fit(values)
            else:
                log.log(logging.DEBUG - 1, f"{rule} matched no columns")

            # collect specialised rule for application
            specialised_rule = copy.copy(rule)
            if not specialised_rule.skip and specialised_rule.independentColumns is None and len(matching_columns) > 1:
                raise ValueError(f"Normalisation rule matching multiple columns {matching_columns} must set `independentColumns` "
                                 f"(got None)")
            specialised_rule.set_regex(or_regex_group(matching_columns))
            self._rules.append(specialised_rule)

    def _check_unhandled_columns(self, df, matched_rules_by_column):
        if self.requireAllHandled:
            unhandled_columns = set(df.columns) - set(matched_rules_by_column.keys())
            if len(unhandled_columns) > 0:
                raise Exception(f"The following columns are not handled by any rules: {unhandled_columns}; "
                                f"rules: {', '.join(map(str, self._rules))}")

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        matched_rules_by_column = {}
        for rule in self._rules:
            matching_columns = rule.matching_columns(df.columns)
            if len(matching_columns) == 0:
                continue
            for c in matching_columns:
                matched_rules_by_column[c] = rule
            if not rule.skip:
                if rule.independentColumns and not rule.arrayValued:
                    matching_columns = sorted(matching_columns)
                    df[matching_columns] = rule.transformer.transform(df[matching_columns].values)
                else:
                    for c in matching_columns:
                        if not rule.arrayValued:
                            df[c] = rule.transformer.transform(df[[c]].values)
                        else:
                            df[c] = [rule.transformer.transform(np.array([x]).T)[:, 0] for x in df[c]]
        self._check_unhandled_columns(df, matched_rules_by_column)
        return df

    def info(self):
        info = super().info()
        info["requireAllHandled"] = self.requireAllHandled
        info["inplace"] = self.inplace
        return info

    def find_rule(self, col_name: str) -> "DFTNormalisation.Rule":
        for rule in self._rules:
            if rule.matches(col_name):
                return rule


class DFTFromColumnGenerators(RuleBasedDataFrameTransformer):
    """
    Extends a data frame with columns generated from ColumnGenerator instances
    """
    def __init__(self, column_generators: Sequence[ColumnGenerator], inplace=False):
        super().__init__()
        self.columnGenerators = column_generators
        self.inplace = inplace

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        for cg in self.columnGenerators:
            series = cg.generate_column(df)
            df[series.name] = series
        return df

    def info(self):
        info = super().info()
        info["inplace"] = self.inplace
        return info


class DFTCountEntries(RuleBasedDataFrameTransformer):
    """
    Transforms a data frame, based on one of its columns, into a new data frame containing two columns that indicate the counts
    of unique values in the input column. It is the "DataFrame output version" of pd.Series.value_counts.
    Each row of the output column holds a unique value of the input column and the number of times it appears in the input column.
    """
    def __init__(self, column_for_entry_count: str, column_name_for_resulting_counts: str = "counts"):
        super().__init__()
        self.columnNameForResultingCounts = column_name_for_resulting_counts
        self.columnForEntryCount = column_for_entry_count

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        series = df[self.columnForEntryCount].value_counts()
        return pd.DataFrame({self.columnForEntryCount: series.index, self.columnNameForResultingCounts: series.values})

    def info(self):
        info = super().info()
        info["columnNameForResultingCounts"] = self.columnNameForResultingCounts
        info["columnForEntryCount"] = self.columnForEntryCount
        return info


class DFTAggregationOnColumn(RuleBasedDataFrameTransformer):
    def __init__(self, column_for_aggregation: str, aggregation: Callable):
        super().__init__()
        self.columnForAggregation = column_for_aggregation
        self.aggregation = aggregation

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(self.columnForAggregation).agg(self.aggregation)


class DFTRoundFloats(RuleBasedDataFrameTransformer):
    def __init__(self, decimals=0):
        super().__init__()
        self.decimals = decimals

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.round(df.values, self.decimals), columns=df.columns, index=df.index)

    def info(self):
        info = super().info()
        info["decimals"] = self.decimals
        return info


class DFTSkLearnTransformer(InvertibleDataFrameTransformer):
    """
    Applies a transformer from sklearn.preprocessing to (a subset of) the columns of a data frame.
    If multiple columns are transformed, they are transformed independently (i.e. each column uses a separately trained transformation).
    """
    def __init__(self,
            sklearn_transformer: SkLearnTransformerProtocol,
            columns: Optional[List[str]] = None,
            inplace=False,
            array_valued=False):
        """
        :param sklearn_transformer: the transformer instance (from sklearn.preprocessing) to use (which will be fitted & applied)
        :param columns: the set of column names to which the transformation shall apply; if None, apply it to all columns
        :param inplace: whether to apply the transformation in-place
        :param array_valued: whether to apply transformation not to scalar-valued columns but to one or more array-valued columns,
            where the values of all arrays within a column (which may vary in length) are to be transformed in the same way.
            If multiple columns are transformed, then the arrays belonging to a single row must all have the same length.
        """
        super().__init__()
        self.set_name(f"{self.__class__.__name__}_wrapped_{sklearn_transformer.__class__.__name__}")
        self.sklearnTransformer = sklearn_transformer
        self.columns = columns
        self.inplace = inplace
        self.arrayValued = array_valued

    def __setstate__(self, state):
        state["arrayValued"] = state.get("arrayValued", False)
        setstate(DFTSkLearnTransformer, self, state)

    def _fit(self, df: pd.DataFrame):
        cols = self.columns
        if cols is None:
            cols = df.columns
        if not self.arrayValued:
            values = df[cols].values
        else:
            if len(cols) == 1:
                values = np.concatenate(df[cols[0]].values.flatten())
                values = values.reshape((len(values), 1))
            else:
                flat_col_arrays = [np.concatenate(df[col].values.flatten()) for col in cols]
                lengths = [len(a) for a in flat_col_arrays]
                if len(set(lengths)) != 1:
                    raise ValueError(f"Columns {cols} do not contain the same number of values: {lengths}")
                values = np.stack(flat_col_arrays, axis=1)
        self.sklearnTransformer.fit(values)

    def _apply_transformer(self, df: pd.DataFrame, inverse: bool) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        cols = self.columns
        if cols is None:
            cols = df.columns
        transform = (lambda x: self.sklearnTransformer.inverse_transform(x)) if inverse else lambda x: self.sklearnTransformer.transform(x)
        if not self.arrayValued:
            df[cols] = transform(df[cols].values)
        else:
            if len(cols) == 1:
                c = cols[0]
                df[c] = [transform(np.array([x]).T)[:, 0] for x in df[c]]
            else:
                transformed_values = [transform(np.stack(row, axis=1)) for row in df.values]
                for iCol, col in enumerate(cols):
                    df[col] = [row[:, iCol] for row in transformed_values]
        return df

    def _apply(self, df):
        return self._apply_transformer(df, False)

    def apply_inverse(self, df):
        return self._apply_transformer(df, True)

    def info(self):
        info = super().info()
        info["columns"] = self.columns
        info["inplace"] = self.inplace
        info["sklearnTransformerClass"] = self.sklearnTransformer.__class__.__name__
        return info


class DFTSortColumns(RuleBasedDataFrameTransformer):
    """
    Sorts a data frame's columns in ascending order
    """
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[sorted(df.columns)]


class DFTFillNA(RuleBasedDataFrameTransformer):
    """
    Fills NA/NaN values with the given value
    """
    def __init__(self, fill_value, inplace: bool = False):
        super().__init__()
        self.fillValue = fill_value
        self.inplace = inplace

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.inplace:
            df.fillna(value=self.fillValue, inplace=True)
            return df
        else:
            return df.fillna(value=self.fillValue)


class DFTCastCategoricalColumns(RuleBasedDataFrameTransformer):
    """
    Casts columns with dtype category to the given type.
    This can be useful in cases where categorical columns are not accepted by the model but the column values are actually numeric,
    in which case the cast to a numeric value yields an acceptable label encoding.
    """
    def __init__(self, columns: Optional[List[str]] = None, dtype=float):
        """
        :param columns: the columns to convert; if None, convert all that have dtype category
        :param dtype: the data type to which categorical columns are to be converted
        """
        super().__init__()
        self.columns = columns
        self.dtype = dtype

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns = self.columns if self.columns is not None else df.columns
        for col in columns:
            s = df[col]
            if s.dtype.name == "category":
                df[col] = s.astype(self.dtype)
        return df
    

class DFTDropNA(RuleBasedDataFrameTransformer):
    """
    Drops rows or columns containing NA/NaN values
    """
    def __init__(self, axis=0, inplace=False):
        """
        :param axis: 0 to drop rows, 1 to drop columns containing an N/A value
        :param inplace: whether to perform the operation in-place on the input data frame
        """
        super().__init__()
        self.axis = axis
        self.inplace = inplace

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.inplace:
            df.dropna(axis=self.axis, inplace=True)
            return df
        else:
            return df.dropna(axis=self.axis)
