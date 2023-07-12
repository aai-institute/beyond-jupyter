import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Sequence, Union, Dict, Callable, Any, Optional, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .sklearn_transformer import SkLearnTransformerProtocol
from ..columngen import ColumnGenerator
from ..util import flattenArguments, countNotNone
from ..util.pandas import DataFrameColumnChangeTracker
from ..util.pickle import setstate
from ..util.string import orRegexGroup, ToStringMixin

from typing import TYPE_CHECKING

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

    def _toStringExcludePrivate(self) -> bool:
        return True

    def getName(self) -> str:
        """
        :return: the name of this dft transformer, which may be a default name if the name has not been set.
        """
        return self._name

    def setName(self, name: str):
        self._name = name

    def withName(self, name: str):
        self.setName(name)
        return self

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._columnChangeTracker = DataFrameColumnChangeTracker(df)
        if not self.isFitted():
            raise Exception(f"Cannot apply a DataFrameTransformer which is not fitted: "
                            f"the df transformer {self.getName()} requires fitting")
        df = self._apply(df)
        self._columnChangeTracker.trackChange(df)
        return df

    def info(self):
        return {
            "name": self.getName(),
            "changeInColumnNames": self._columnChangeTracker.columnChangeString() if self._columnChangeTracker is not None else None,
            "isFitted": self.isFitted(),
        }

    def fit(self, df: pd.DataFrame):
        self._fit(df)
        self._isFitted = True

    def isFitted(self):
        return self._isFitted

    def fitApply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.apply(df)

    def toFeatureGenerator(self, categoricalFeatureNames: Optional[Union[Sequence[str], str]] = None,
            normalisationRules: Sequence['DFTNormalisation.Rule'] = (),
            normalisationRuleTemplate: 'DFTNormalisation.RuleTemplate' = None,
            addCategoricalDefaultRules=True):
        # need to import here to prevent circular imports
        from ..featuregen import FeatureGeneratorFromDFT
        return FeatureGeneratorFromDFT(
            self, categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules,
            normalisationRuleTemplate=normalisationRuleTemplate, addCategoricalDefaultRules=addCategoricalDefaultRules
        )


class DFTFromFeatureGenerator(DataFrameTransformer):
    def _fit(self, df: pd.DataFrame):
        self.fgen.fit(df, ctx=None)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fgen.generate(df)

    def __init__(self, fgen: "FeatureGenerator"):
        super().__init__()
        self.fgen = fgen
        self.setName(f"{self.__class__.__name__}[{self.fgen.getName()}]")


class InvertibleDataFrameTransformer(DataFrameTransformer, ABC):
    @abstractmethod
    def applyInverse(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def getInverse(self) -> "InverseDataFrameTransformer":
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

    def isFitted(self):
        return True


class InverseDataFrameTransformer(RuleBasedDataFrameTransformer):
    def __init__(self, invertibleDFT: InvertibleDataFrameTransformer):
        super().__init__()
        self.invertibleDFT = invertibleDFT

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.invertibleDFT.applyInverse(df)


class DataFrameTransformerChain(DataFrameTransformer):
    """
    Supports the application of a chain of data frame transformers.
    During fit and apply each transformer in the chain receives the transformed output of its predecessor.
    """

    def __init__(self, *dataFrameTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]):
        super().__init__()
        self.dataFrameTransformers = flattenArguments(dataFrameTransformers)

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
            df = transformer.fitApply(df)
        self.dataFrameTransformers[-1].fit(df)

    def isFitted(self):
        return all([dft.isFitted() for dft in self.dataFrameTransformers])

    def getNames(self) -> List[str]:
        """
        :return: the list of names of all contained feature generators
        """
        return [transf.getName() for transf in self.dataFrameTransformers]

    def info(self):
        info = super().info()
        info["chainedDFTTransformerNames"] = self.getNames()
        info["length"] = len(self)
        return info

    def findFirstTransformerByType(self, cls) -> Optional[DataFrameTransformer]:
        for dft in self.dataFrameTransformers:
            if isinstance(dft, cls):
                return dft
        return None


class DFTRenameColumns(RuleBasedDataFrameTransformer):
    def __init__(self, columnsMap: Dict[str, str]):
        """
        :param columnsMap: dictionary mapping old column names to new names
        """
        super().__init__()
        self.columnsMap = columnsMap

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
    def __init__(self, column: str, setToKeep: Set):
        super().__init__()
        self.setToKeep = setToKeep
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
    def __init__(self, column: str, setToDrop: Set):
        super().__init__()
        self.setToDrop = setToDrop
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
    def __init__(self, column: str, vectorizedCondition: Callable[[pd.Series], Sequence[bool]]):
        super().__init__()
        self.column = column
        self.vectorizedCondition = vectorizedCondition

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
    def __init__(self, column: str, columnTransform: Union[Callable, np.ufunc]):
        """
        :param column: the name of the column to be modified
        :param columnTransform: a function operating on single cells or a Numpy ufunc that applies to an entire Series
        """
        super().__init__()
        self.column = column
        self.columnTransform = columnTransform

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = df[self.column].apply(self.columnTransform)
        return df


class DFTModifyColumnVectorized(RuleBasedDataFrameTransformer):
    """
    Modifies a column specified by 'column' using 'columnTransform'. This transformer can be used to utilise Numpy vectorisation for
    performance optimisation.
    """
    def __init__(self, column: str, columnTransform: Callable[[np.ndarray], Union[Sequence, pd.Series, np.ndarray]]):
        """
        :param column: the name of the column to be modified
        :param columnTransform: a function that takes a Numpy array and from which the returned value will be assigned to the column as a whole
        """
        super().__init__()
        self.column = column
        self.columnTransform = columnTransform

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = self.columnTransform(df[self.column].values)
        return df


class DFTOneHotEncoder(DataFrameTransformer):
    def __init__(self, columns: Optional[Union[str, Sequence[str]]],
            categories: Union[List[np.ndarray], Dict[str, np.ndarray]] = None, inplace=False, ignoreUnknown=False,
            arrayValuedResult=False):
        """
        One hot encode categorical variables

        :param columns: list of names or regex matching names of columns that are to be replaced by a list one-hot encoded columns each
            (or an array-valued column for the case where useArrayValues=True);
            If None, then no columns are actually to be one-hot-encoded
        :param categories: numpy arrays containing the possible values of each of the specified columns (for case where sequence is specified
            in 'columns') or dictionary mapping column name to array of possible categories for the column name.
            If None, the possible values will be inferred from the columns
        :param inplace: whether to perform the transformation in-place
        :param ignoreUnknown: if True and an unknown category is encountered during transform, the resulting one-hot
            encoded columns for this feature will be all zeros. if False, an unknown category will raise an error.
        :param arrayValuedResult: whether to replace the input columns by columns of the same name containing arrays as values
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
            self._columnNameRegex = orRegexGroup(columns)
            self._columnsToEncode = columns
        self.inplace = inplace
        self.arrayValuedResult = arrayValuedResult
        self.handleUnknown = "ignore" if ignoreUnknown else "error"
        if categories is not None:
            if type(categories) == dict:
                self.oneHotEncoders = {col: OneHotEncoder(categories=[np.sort(categories)], sparse=False, handle_unknown=self.handleUnknown) for col, categories in categories.items()}
            else:
                if len(columns) != len(categories):
                    raise ValueError(f"Given categories must have the same length as columns to process")
                self.oneHotEncoders = {col: OneHotEncoder(categories=[np.sort(categories)], sparse=False, handle_unknown=self.handleUnknown) for col, categories in zip(columns, categories)}

    def __setstate__(self, state):
        if "arrayValuedResult" not in state:
            state["arrayValuedResult"] = False
        super().__setstate__(state)

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        d = super()._toStringAdditionalEntries()
        d["columns"] = self._paramInfo.get("columns")
        return d

    def _fit(self, df: pd.DataFrame):
        if self._columnsToEncode is None:
            self._columnsToEncode = [c for c in df.columns if re.fullmatch(self._columnNameRegex, c) is not None]
            if len(self._columnsToEncode) == 0:
                log.warning(f"{self} does not apply to any columns, transformer has no effect; regex='{self._columnNameRegex}'")
        if self.oneHotEncoders is None:
            self.oneHotEncoders = {column: OneHotEncoder(categories=[np.sort(df[column].unique())], sparse=False, handle_unknown=self.handleUnknown) for column in self._columnsToEncode}
        for columnName in self._columnsToEncode:
            self.oneHotEncoders[columnName].fit(df[[columnName]])

    def _apply(self, df: pd.DataFrame):
        if len(self._columnsToEncode) == 0:
            return df

        if not self.inplace:
            df = df.copy()
        for columnName in self._columnsToEncode:
            encodedArray = self.oneHotEncoders[columnName].transform(df[[columnName]])
            if not self.arrayValuedResult:
                df = df.drop(columns=columnName)
                for i in range(encodedArray.shape[1]):
                    df["%s_%d" % (columnName, i)] = encodedArray[:, i]
            else:
                df[columnName] = list(encodedArray)
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
        def __init__(self, skip=False, unsupported=False, transformer: SkLearnTransformerProtocol = None,
                transformerFactory: Callable[[], SkLearnTransformerProtocol] = None, independentColumns: Optional[bool] = None):
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

            :param skip: flag indicating whether no transformation shall be performed on all of the columns (because they are already normalised)
            :param unsupported: flag indicating whether normalisation of all columns is unsupported (shall trigger an exception if attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to the matching column(s)
                for the case where a transformation is necessary (skip=False, unsupported=False). If None is given, either transformerFactory
                or the containing instance's default factory will be used.
                NOTE: Use an instance only if you want, in particular, the instance to be shared across several models that use the same
                feature with associated rule/rule template (disabling `fit` where appropriate). Otherwise, use a factory.
            :param transformerFactory: a factory for the generation of the transformer instance, which will only be applied if `transformer`
                is not given; if neither `transformer` nor `transformerInstance` are given, the containing instance's default factory will
                be used. See :class:`SkLearnTransformerFactoryFactory` for convenient construction options.
            :param independentColumns: whether, for the case where the rule matches multiple columns, the columns are independent and a separate transformation
                is to be learned for each of them (rather than using the same transformation for all columns and learning the transformation from the data
                of all columns); must be specified for rules matching more than one column, None is acceptable only for a single column
            """
            if (skip or unsupported) and countNotNone(transformer, transformerFactory) > 0:
                raise ValueError("Passed transformer or transformerFactory while skip=True or unsupported=True")
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer
            self.transformerFactory = transformerFactory
            self.independentColumns = independentColumns

        def toRule(self, regex: Optional[str]):
            """
            Convert the template to a rule for all columns matching the regex

            :param regex: a regular expression defining the column the rule applies to
            :return: the resulting Rule
            """
            return DFTNormalisation.Rule(regex, skip=self.skip, unsupported=self.unsupported, transformer=self.transformer,
                transformerFactory=self.transformerFactory, independentColumns=self.independentColumns)

        def toPlaceholderRule(self):
            return self.toRule(None)

    class Rule(ToStringMixin):
        def __init__(self, regex: Optional[str], skip=False, unsupported=False, transformer: SkLearnTransformerProtocol = None,
                transformerFactory: Callable[[], SkLearnTransformerProtocol] = None,
                arrayValued=False, fit=True, independentColumns: Optional[bool] = None):
            """
            :param regex: a regular expression defining the column(s) the rule applies to.
                If it applies to multiple columns, these columns will be normalised in the same way (using the same normalisation
                process for each column) unless independentColumns=True.
                If None, the rule is a placeholder rule and the regex must be set later via setRegex or the rule will not be applicable.
            :param skip: flag indicating whether no transformation shall be performed on the matching column(s)
            :param unsupported: flag indicating whether normalisation of the matching column(s) is unsupported (shall trigger an exception if attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to the matching column(s)
                for the case where a transformation is necessary (skip=False, unsupported=False). If None is given, either transformerFactory
                or the containing instance's default factory will be used.
                NOTE: Use an instance only if you want, in particular, the instance to be shared across several models that use the same
                feature with associated rule/rule template (disabling `fit` where appropriate). Otherwise, use a factory.
            :param transformerFactory: a factory for the generation of the transformer instance, which will only be applied if `transformer`
                is not given; if neither `transformer` nor `transformerInstance` are given, the containing instance's default factory will
                be used. See :class:`SkLearnTransformerFactoryFactory` for convenient construction options.
            :param arrayValued: whether the column values are not scalars but arrays (of arbitrary lengths).
                It is assumed that all entries in such arrays are to be normalised in the same way.
                If arrayValued is True, only a single matching column is supported, i.e. the regex must match at most one column.
            :param fit: whether the rule's transformer shall be fitted
            :param independentColumns: whether, for the case where the rule matches multiple columns, the columns are independent and a separate transformation
                is to be learned for each of them (rather than using the same transformation for all columns and learning the transformation from the data
                of all columns); must be specified for rules matching more than one column, None is acceptable only for single-column rules
            """
            if skip and (transformer is not None or transformerFactory is not None):
                raise ValueError("skip==True while transformer/transformerFactory is not None")
            self.regex = re.compile(regex) if regex is not None else None
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer
            self.transformerFactory = transformerFactory
            self.arrayValued = arrayValued
            self.fit = fit
            self.independentColumns = independentColumns

        def __setstate__(self, state):
            setstate(DFTNormalisation.Rule, self, state, newDefaultProperties=dict(arrayValued=False, fit=True, independentColumns=False,
                    transformerFactory=None))

        def _toStringExcludes(self) -> List[str]:
            return super()._toStringExcludes() + ["regex"]

        def _toStringAdditionalEntries(self) -> Dict[str, Any]:
            d = super()._toStringAdditionalEntries()
            if self.regex is not None:
                d["regex"] = f"'{self.regex.pattern}'"
            return d

        def setRegex(self, regex: str):
            try:
                self.regex = re.compile(regex)
            except Exception as e:
                raise Exception(f"Could not compile regex '{regex}': {e}")

        def matches(self, column: str):
            if self.regex is None:
                raise Exception("Attempted to apply a placeholder rule. Perhaps the feature generator from which the rule originated was never applied in order to have the rule instantiated.")
            return self.regex.fullmatch(column) is not None

        def matchingColumns(self, columns: Sequence[str]) -> List[str]:
            return [col for col in columns if self.matches(col)]

    def __init__(self, rules: Sequence[Rule], defaultTransformerFactory=None, requireAllHandled=True, inplace=False):
        """
        :param rules: the set of rules; rules are always fitted and applied in the given order.
            A convenient way to obtain a set of rules in the :class:`sensai.vector_model.VectorModel` context is from a
            :class:`sensai.featuregen.FeatureCollector` or :class:`sensai.featuregen.MultiFeatureGenerator`.
        :param defaultTransformerFactory: a factory for the creation of transformer instances (from sklearn.preprocessing, e.g. StandardScaler)
            that shall be used to create a transformer for all rules that don't specify a particular transformer.
            The default transformer will only be applied to columns matched by such rules, unmatched columns will
            not be transformed.
        :param requireAllHandled: whether to raise an exception if not all columns are matched by a rule
        :param inplace: whether to apply data frame transformations in-place
        """
        super().__init__()
        self.requireAllHandled = requireAllHandled
        self.inplace = inplace
        self._userRules = rules
        self._defaultTransformerFactory = defaultTransformerFactory
        self._rules = None

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        d = super()._toStringAdditionalEntries()
        if self._rules is not None:
            d["rules"] = self._rules
        else:
            d["userRules"] = self._userRules
        return d

    def _fit(self, df: pd.DataFrame):
        matchedRulesByColumn = {}
        self._rules = []
        for rule in self._userRules:
            matchingColumns = rule.matchingColumns(df.columns)
            for c in matchingColumns:
                if c in matchedRulesByColumn:
                    raise Exception(f"More than one rule applies to column '{c}': {matchedRulesByColumn[c]}, {rule}")
                matchedRulesByColumn[c] = rule

            if len(matchingColumns) > 0:
                if rule.unsupported:
                    raise Exception(f"Normalisation of columns {matchingColumns} is unsupported according to {rule}. If you want to make use of these columns, transform them into a supported column before applying {self.__class__.__name__}.")
                if not rule.skip:
                    if rule.transformer is None:
                        if rule.transformerFactory is not None:
                            rule.transformer = rule.transformerFactory()
                        else:
                            if self._defaultTransformerFactory is None:
                                raise Exception(f"No transformer to fit: {rule} defines no transformer and instance has no transformer factory")
                            rule.transformer = self._defaultTransformerFactory()
                    if rule.fit:
                        # fit transformer
                        applicableDF = df[sorted(matchingColumns)]
                        if rule.arrayValued:
                            if len(matchingColumns) > 1:
                                raise Exception(f"Array-valued case is only supported for a single column, matched {matchingColumns} for {rule}")
                            values = np.concatenate(applicableDF.values.flatten())
                            values = values.reshape((len(values), 1))
                        elif rule.independentColumns:
                            values = applicableDF.values
                        else:
                            values = applicableDF.values.flatten()
                            values = values.reshape((len(values), 1))
                        rule.transformer.fit(values)
            else:
                log.log(logging.DEBUG - 1, f"{rule} matched no columns")

            # collect specialised rule for application
            specialisedRule = copy.copy(rule)
            if not specialisedRule.skip and specialisedRule.independentColumns is None and len(matchingColumns) > 1:
                raise ValueError(f"Normalisation rule matching multiple columns {matchingColumns} must set `independentColumns` (got None)")
            specialisedRule.setRegex(orRegexGroup(matchingColumns))
            self._rules.append(specialisedRule)

    def _checkUnhandledColumns(self, df, matchedRulesByColumn):
        if self.requireAllHandled:
            unhandledColumns = set(df.columns) - set(matchedRulesByColumn.keys())
            if len(unhandledColumns) > 0:
                raise Exception(f"The following columns are not handled by any rules: {unhandledColumns}; rules: {', '.join(map(str, self._rules))}")

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        matchedRulesByColumn = {}
        for rule in self._rules:
            matchingColumns = rule.matchingColumns(df.columns)
            if len(matchingColumns) == 0:
                continue
            for c in matchingColumns:
                matchedRulesByColumn[c] = rule
            if not rule.skip:
                if rule.independentColumns and not rule.arrayValued:
                    matchingColumns = sorted(matchingColumns)
                    df[matchingColumns] = rule.transformer.transform(df[matchingColumns].values)
                else:
                    for c in matchingColumns:
                        if not rule.arrayValued:
                            df[c] = rule.transformer.transform(df[[c]].values)
                        else:
                            df[c] = [rule.transformer.transform(np.array([x]).T)[:, 0] for x in df[c]]
        self._checkUnhandledColumns(df, matchedRulesByColumn)
        return df

    def info(self):
        info = super().info()
        info["requireAllHandled"] = self.requireAllHandled
        info["inplace"] = self.inplace
        return info

    def findRule(self, colName: str) -> "DFTNormalisation.Rule":
        for rule in self._rules:
            if rule.matches(colName):
                return rule


class DFTFromColumnGenerators(RuleBasedDataFrameTransformer):
    """
    Extends a data frame with columns generated from ColumnGenerator instances
    """
    def __init__(self, columnGenerators: Sequence[ColumnGenerator], inplace=False):
        super().__init__()
        self.columnGenerators = columnGenerators
        self.inplace = inplace

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        for cg in self.columnGenerators:
            series = cg.generateColumn(df)
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
    def __init__(self, columnForEntryCount: str, columnNameForResultingCounts: str = "counts"):
        super().__init__()
        self.columnNameForResultingCounts = columnNameForResultingCounts
        self.columnForEntryCount = columnForEntryCount

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        series = df[self.columnForEntryCount].value_counts()
        return pd.DataFrame({self.columnForEntryCount: series.index, self.columnNameForResultingCounts: series.values})

    def info(self):
        info = super().info()
        info["columnNameForResultingCounts"] = self.columnNameForResultingCounts
        info["columnForEntryCount"] = self.columnForEntryCount
        return info


class DFTAggregationOnColumn(RuleBasedDataFrameTransformer):
    def __init__(self, columnForAggregation: str, aggregation: Callable):
        super().__init__()
        self.columnForAggregation = columnForAggregation
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
    Applies a transformer from sklearn.preprocessing to (a subset of the columns of) a data frame.
    If multiple columns are transformed, they are transformed independently (i.e. each column uses a separately trained transformation).
    """
    def __init__(self, sklearnTransformer: SkLearnTransformerProtocol, columns: Optional[List[str]] = None, inplace=False,
            arrayValued=False):
        """
        :param sklearnTransformer: the transformer instance (from sklearn.preprocessing) to use (which will be fitted & applied)
        :param columns: the set of column names to which the transformation shall apply; if None, apply it to all columns
        :param inplace: whether to apply the transformation in-place
        :param arrayValued: whether to apply transformation not to scalar-valued columns but to one or more array-valued columns,
            where the values of all arrays within a column (which may vary in length) are to be transformed in the same way.
            If multiple columns are transformed, then the arrays belonging to a single row must all have the same length.
        """
        super().__init__()
        self.setName(f"{self.__class__.__name__}_wrapped_{sklearnTransformer.__class__.__name__}")
        self.sklearnTransformer = sklearnTransformer
        self.columns = columns
        self.inplace = inplace
        self.arrayValued = arrayValued

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
                flatColArrays = [np.concatenate(df[col].values.flatten()) for col in cols]
                lengths = [len(a) for a in flatColArrays]
                if len(set(lengths)) != 1:
                    raise ValueError(f"Columns {cols} do not contain the same number of values: {lengths}")
                values = np.stack(flatColArrays, axis=1)
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
                transformedValues = [transform(np.stack(row, axis=1)) for row in df.values]
                for iCol, col in enumerate(cols):
                    df[col] = [row[:, iCol] for row in transformedValues]
        return df

    def _apply(self, df):
        return self._apply_transformer(df, False)

    def applyInverse(self, df):
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
    def __init__(self, fillValue, inplace: bool = False):
        super().__init__()
        self.fillValue = fillValue
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
