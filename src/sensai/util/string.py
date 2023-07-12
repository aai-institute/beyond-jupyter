import functools
import re
import sys
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Sequence, Iterable, Optional, Mapping, Callable

reCommaWhitespacePotentiallyBreaks = re.compile(r",\s+")


class StringConverter(ABC):
    """
    Abstraction for a string conversion mechanism
    """
    @abstractmethod
    def toString(self, x) -> str:
        pass


def dictString(d: Mapping, brackets: Optional[str] = None, converter: StringConverter = None):
    """
    Converts a dictionary to a string of the form "<key>=<value>, <key>=<value>, ...", optionally enclosed
    by brackets

    :param d: the dictionary
    :param brackets: a two-character string containing the opening and closing bracket to use, e.g. ``"{}"``;
        if None, do not use enclosing brackets
    :param converter: the string converter to use for values
    :return: the string representation
    """
    s = ', '.join([f'{k}={toString(v, converter=converter)}' for k, v in d.items()])
    if brackets is not None:
        return brackets[:1] + s + brackets[-1:]
    else:
        return s


def listString(l: Iterable[Any], brackets="[]", quote: Optional[str] = None, converter: StringConverter = None):
    """
    Converts a list or any other iterable to a string of the form "[<value>, <value>, ...]", optionally enclosed
    by different brackets or with the values quoted.

    :param d: the dictionary
    :param brackets: a two-character string containing the opening and closing bracket to use, e.g. ``"[]"``;
        if None, do not use enclosing brackets
    :param quote: a 1-character string defining the quote to use around each value, e.g. ``"'"``.
    :param converter: the string converter to use for values
    :return: the string representation
    """
    def item(x):
        x = toString(x, converter=converter)
        if quote is not None:
            return quote + x + quote
        else:
            return x
    s = ", ".join((item(x) for x in l))
    if brackets is not None:
        return brackets[:1] + s + brackets[-1:]
    else:
        return s


def toString(x, converter: StringConverter = None, applyConverterToNonComplexObjects=True):
    """
    Converts the given object to a string, with proper handling of lists, tuples and dictionaries, optionally using a converter.
    The conversion also removes unwanted line breaks (as present, in particular, in sklearn's string representations).

    :param x: the object to convert
    :param converter: the converter with which to convert objects to strings
    :param applyConverterToNonComplexObjects: whether to apply/pass on the converter (if any) not only when converting complex objects but also
        non-complex, primitive objects; use of this flag enables converters to implement their conversion functionality using this function
        for complex objects without causing an infinite recursion.
    :return: the string representation
    """
    if type(x) == list:
        return listString(x, converter=converter)
    elif type(x) == tuple:
        return listString(x, brackets="()", converter=converter)
    elif type(x) == dict:
        return dictString(x, brackets="{}", converter=converter)
    else:
        if converter and applyConverterToNonComplexObjects:
            s = converter.toString(x)
        else:
            s = str(x)
        s = reCommaWhitespacePotentiallyBreaks.sub(", ", s)  # remove any unwanted line breaks and indentation after commas (as generated, for example, by sklearn objects)
        return s


def objectRepr(obj, memberNamesOrDict: Union[List[str], Dict[str, Any]]):
    if type(memberNamesOrDict) == dict:
        membersDict = memberNamesOrDict
    else:
        membersDict = {m: toString(getattr(obj, m)) for m in memberNamesOrDict}
    return f"{obj.__class__.__name__}[{dictString(membersDict)}]"


def orRegexGroup(allowedNames: Sequence[str]):
    """

    :param allowedNames: strings to include as literals in the regex
    :return: a regular expression string of the form (<name1>| ...|<nameN>), which any of the given names
    """
    allowedNames = [re.escape(name) for name in allowedNames]
    return r"(%s)" % "|".join(allowedNames)


def functionName(x: Callable) -> str:
    if isinstance(x, functools.partial):
        return functionName(x.func)
    elif hasattr(x, "__name__"):
        return x.__name__
    else:
        return str(x)


# TODO: allow returning json string for easier parsing/printing
class ToStringMixin:
    """
    Provides implementations for ``__str__`` and ``__repr__`` which are based on the format ``"<class name>[<object info>]"`` and
    ``"<class name>[id=<object id>, <object info>]"`` respectively, where ``<object info>`` is usually a list of entries of the
    form ``"<name>=<value>, ..."``.

    By default, ``<class name>`` will be the qualified name of the class, and ``<object info>`` will include all properties
    of the class, including private ones starting with an underscore (though the underscore will be dropped in the string
    representation).

        * To exclude private properties, override :meth:`_toStringExcludePrivate` to return True. If there are exceptions
          (and some private properties shall be retained), additionally override :meth:`_toStringExcludeExceptions`.
        * To exclude a particular set of properties, override :meth:`_toStringExcludes`.
        * To include only select properties (introducing inclusion semantics), override :meth:`_toStringIncludes`.
        * To add values to the properties list that aren't actually properties of the object (i.e. derived properties),
          override :meth:`_toStringAdditionalEntries`.
        * To define a fully custom representation for ``<object info>`` which is not based on the above principles, override
          :meth:`_toStringObjectInfo`.

    For well-defined string conversions within a class hierarchy, it can be a good practice to define additional
    inclusions/exclusions by overriding the respective method once more and basing the return value on an extended
    version of the value returned by superclass.
    In some cases, the requirements of a subclass can be at odds with the definitions in the superclass: The superclass
    may make use of exclusion semantics, but the subclass may want to use inclusion semantics (and include
    only some of the many properties it adds). In this case, if the subclass used :meth:`_toStringInclude`, the exclusion semantics
    of the superclass would be void and none of its properties would actually be included.
    In such cases, override :meth:`_toStringIncludesForced` to add inclusions regardless of the semantics otherwise used along
    the class hierarchy.

    .. document private functions
    .. automethod:: _toStringClassName
    .. automethod:: _toStringObjectInfo
    .. automethod:: _toStringExcludes
    .. automethod:: _toStringExcludeExceptions
    .. automethod:: _toStringIncludes
    .. automethod:: _toStringIncludesForced
    .. automethod:: _toStringAdditionalEntries
    .. automethod:: _toStringExcludePrivate
    """
    _TOSTRING_INCLUDE_ALL = "__all__"

    def _toStringClassName(self):
        """
        :return: the string use for <class name> in the string representation ``"<class name>[<object info]"``
        """
        return type(self).__qualname__

    def _toStringProperties(self, exclude: Optional[Union[str, Iterable[str]]] = None, include: Optional[Union[str, Iterable[str]]] = None,
            excludeExceptions: Optional[List[str]] = None, includeForced: Optional[List[str]] = None,
            additionalEntries: Dict[str, Any] = None, converter: StringConverter = None) -> str:
        """
        Creates a string of the class attributes, with optional exclusions/inclusions/additions.
        Exclusions take precedence over inclusions.

        :param exclude: attributes to be excluded
        :param include: attributes to be included; if non-empty, only the specified attributes will be printed (bar the ones
            excluded by ``exclude``)
        :param includeForced: additional attributes to be included
        :param additionalEntries: additional key-value entries to be added
        :param converter: the string converter to use; if None, use default (which avoids infinite recursions)
        :return: a string containing entry/property names and values
        """
        def mklist(x):
            if x is None:
                return []
            if type(x) == str:
                return [x]
            return x

        exclude = mklist(exclude)
        include = mklist(include)
        includeForced = mklist(includeForced)
        excludeExceptions = mklist(excludeExceptions)

        def isExcluded(k):
            if k in includeForced or k in excludeExceptions:
                return False
            if k in exclude:
                return True
            if self._toStringExcludePrivate():
                isPrivate = k.startswith("_")
                return isPrivate
            else:
                return False

        # determine relevant attribute dictionary
        if len(include) == 1 and include[0] == self._TOSTRING_INCLUDE_ALL:  # exclude semantics (include everything by default)
            attributeDict = self.__dict__
        else:  # include semantics (include only inclusions)
            attributeDict = {k: getattr(self, k) for k in set(include + includeForced) if hasattr(self, k) and k != self._TOSTRING_INCLUDE_ALL}

        # apply exclusions and remove underscores from attribute names
        d = {k.strip("_"): v for k, v in attributeDict.items() if not isExcluded(k)}

        if additionalEntries is not None:
            d.update(additionalEntries)

        if converter is None:
            converter = self._StringConverterAvoidToStringMixinRecursion(self)
        return dictString(d, converter=converter)

    def _toStringObjectInfo(self) -> str:
        """
        Override this method to use a fully custom definition of the ``<object info>`` part in the full string
        representation ``"<class name>[<object info>]"`` to be generated.
        As soon as this method is overridden, any property-based exclusions, inclusions, etc. will have no effect
        (unless the implementation is specifically designed to make use of them - as is the default
        implementation).
        NOTE: Overrides must not internally use super() because of a technical limitation in the proxy
        object that is used for nested object structures.

        :return: a string containing the string to use for ``<object info>``
        """
        return self._toStringProperties(exclude=self._toStringExcludes(), include=self._toStringIncludes(),
            excludeExceptions=self._toStringExcludeExceptions(), includeForced=self._toStringIncludesForced(),
            additionalEntries=self._toStringAdditionalEntries())

    def _toStringExcludes(self) -> List[str]:
        """
        Makes the string representation exclude the returned attributes.
        This method can be conveniently overridden by subclasses which can call super and extend the list returned.

        This method will only have no effect if :meth:`_toStringObjectInfo` is overridden to not use its result.

        :return: a list of attribute names
        """
        return []

    def _toStringIncludes(self) -> List[str]:
        """
        Makes the string representation include only the returned attributes (i.e. introduces inclusion semantics);
        By default, the list contains only a marker element, which is interpreted as "all attributes included".

        This method can be conveniently overridden by sub-classes which can call super and extend the list returned.
        Note that it is not a problem for a list containing the aforementioned marker element (which stands for all attributes)
        to be extended; the marker element will be ignored and only the user-added elements will be considered as included.

        Note: To add an included attribute in a sub-class, regardless of any super-classes using exclusion or inclusion semantics,
        use _toStringIncludesForced instead.

        This method will have no effect if :meth:`_toStringObjectInfo` is overridden to not use its result.

        :return: a list of attribute names to be included in the string representation
        """
        return [self._TOSTRING_INCLUDE_ALL]

    # noinspection PyMethodMayBeStatic
    def _toStringIncludesForced(self) -> List[str]:
        """
        Defines a list of attribute names that are required to be present in the string representation, regardless of the
        instance using include semantics or exclude semantics, thus facilitating added inclusions in sub-classes.

        This method will have no effect if :meth:`_toStringObjectInfo` is overridden to not use its result.

        :return: a list of attribute names
        """
        return []

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        """
        :return: a dictionary of entries to be included in the ``<object info>`` part of the string representation
        """
        return {}

    def _toStringExcludePrivate(self) -> bool:
        """
        :return: whether to exclude properties that are private (start with an underscore); explicitly included attributes
            will still be considered - as will properties exempt from the rule via :meth:`toStringExcludeException`.
        """
        return False

    def _toStringExcludeExceptions(self) -> List[str]:
        """
        Defines attribute names which should not be excluded even though other rules (particularly the exclusion of private members
        via :meth:`_toStringExcludePrivate`) would otherwise exclude them.

        :return: a list of attribute names
        """
        return []

    def __str__(self):
        return f"{self._toStringClassName()}[{self._toStringObjectInfo()}]"

    def __repr__(self):
        info = f"id={id(self)}"
        propertyInfo = self._toStringObjectInfo()
        if len(propertyInfo) > 0:
            info += ", " + propertyInfo
        return f"{self._toStringClassName()}[{info}]"

    def pprint(self, file=sys.stdout):
        """
        Prints a prettily formatted string representation of the object (with line breaks and indentations)
        to ``stdout`` or the given file.

        :param file: the file to print to
        """
        print(self.pprints(), file=file)

    def pprints(self) -> str:
        """
        :return: a prettily formatted string representation with line breaks and indentations
        """
        return prettyStringRepr(self)

    class _StringConverterAvoidToStringMixinRecursion(StringConverter):
        """
        Avoids recursions when converting objects implementing :class:`ToStringMixin` which may contain themselves to strings.
        Use of this object prevents infinite recursions caused by a :class:`ToStringMixin` instance recursively containing itself in
        either a property of another :class:`ToStringMixin`, a list or a tuple.
        It handles all :class:`ToStringMixin` instances recursively encountered.

        A previously handled instance is converted to a string of the form "<class name>[<<]".
        """
        def __init__(self, *handledObjects: "ToStringMixin"):
            """
            :param handledObjects: objects which are initially assumed to have been handled already
            """
            self._handledToStringMixinIds = set([id(o) for o in handledObjects])

        def toString(self, x) -> str:
            if isinstance(x, ToStringMixin):
                oid = id(x)
                if oid in self._handledToStringMixinIds:
                    return f"{x._toStringClassName()}[<<]"
                self._handledToStringMixinIds.add(oid)
                return str(self._ToStringMixinProxy(x, self))
            else:
                return toString(x, converter=self, applyConverterToNonComplexObjects=False)

        class _ToStringMixinProxy:
            """
            A proxy object which wraps a ToStringMixin to ensure that the converter is applied when creating the properties string.
            The proxy is to achieve that all ToStringMixin methods that aren't explicitly overwritten are bound to this proxy
            (rather than the original object), such that the transitive call to _toStringProperties will call the new
            implementation.
            """

            # methods where we assume that they could transitively call _toStringProperties (others are assumed not to)
            TOSTRING_METHODS_TRANSITIVELY_CALLING_TOSTRINGPROPERTIES = set(["_toStringObjectInfo"])

            def __init__(self, x: "ToStringMixin", converter):
                self.x = x
                self.converter = converter

            def _toStringProperties(self, *args, **kwargs):
                return self.x._toStringProperties(*args, **kwargs, converter=self.converter)

            def _toStringClassName(self):
                return self.x._toStringClassName()

            def __getattr__(self, attr: str):
                if attr.startswith("_toString"):  # ToStringMixin method which we may bind to use this proxy to ensure correct transitive call
                    method = getattr(self.x.__class__, attr)
                    obj = self if attr in self.TOSTRING_METHODS_TRANSITIVELY_CALLING_TOSTRINGPROPERTIES else self.x
                    return lambda *args, **kwargs: method(obj, *args, **kwargs)
                else:
                    return getattr(self.x, attr)

            def __str__(self: "ToStringMixin"):
                return ToStringMixin.__str__(self)


def prettyStringRepr(s: Any, initialIndentationLevel=0, indentationString="    "):
    """
    Creates a pretty string representation (using indentations) from the given object/string representation (as generated, for example, via
    ToStringMixin). An indentation level is added for every opening bracket.

    :param s: an object or object string representation
    :param initialIndentationLevel: the initial indentation level
    :param indentationString: the string which corresponds to a single indentation level
    :return: a reformatted version of the input string with added indentations and line breaks
    """
    if type(s) != str:
        s = str(s)
    indent = initialIndentationLevel
    result = indentationString * indent
    i = 0

    def nl():
        nonlocal result
        result += "\n" + (indentationString * indent)

    def take(cnt=1):
        nonlocal result, i
        result += s[i:i+cnt]
        i += cnt

    def findMatching(j):
        start = j
        op = s[j]
        cl = {"[": "]", "(": ")", "'": "'"}[s[j]]
        isBracket = cl != s[j]
        stack = 0
        while j < len(s):
            if s[j] == op and (isBracket or j == start):
                stack += 1
            elif s[j] == cl:
                stack -= 1
            if stack == 0:
                return j
            j += 1
        return None

    brackets = "[("
    quotes = "'"
    while i < len(s):
        isBracket = s[i] in brackets
        isQuote = s[i] in quotes
        if isBracket or isQuote:
            iMatch = findMatching(i)
            takeFullMatchWithoutBreak = False
            if iMatch is not None:
                k = iMatch + 1
                fullMatch = s[i:k]
                takeFullMatchWithoutBreak = isQuote or not("=" in fullMatch and "," in fullMatch)
                if takeFullMatchWithoutBreak:
                    take(k-i)
            if not takeFullMatchWithoutBreak:
                take(1)
                indent += 1
                nl()
        elif s[i] in "])":
            take(1)
            indent -= 1
        elif s[i:i+2] == ", ":
            take(2)
            nl()
        else:
            take(1)

    return result