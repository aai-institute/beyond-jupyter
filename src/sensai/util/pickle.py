import logging
import os
import pickle
from typing import List, Dict, Any, Iterable

import joblib

from .io import isS3Path, S3Object

log = logging.getLogger(__name__)


def loadPickle(path, backend="pickle"):
    def readFile(f):
        if backend == "pickle":
            try:
                return pickle.load(f)
            except:
                log.error(f"Error loading {path}")
                raise
        elif backend == "joblib":
            return joblib.load(f)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    if isS3Path(path):
        return readFile(S3Object(path).openFile("rb"))
    with open(path, "rb") as f:
        return readFile(f)


def dumpPickle(obj, picklePath, backend="pickle", protocol=pickle.HIGHEST_PROTOCOL):
    def openFile():
        if isS3Path(picklePath):
            return S3Object(picklePath).openFile("wb")
        else:
            return open(picklePath, "wb")

    dirName = os.path.dirname(picklePath)
    if dirName != "":
        os.makedirs(dirName, exist_ok=True)
    with openFile() as f:
        if backend == "pickle":
            try:
                pickle.dump(obj, f, protocol=protocol)
            except AttributeError as e:
                failingPaths = PickleFailureDebugger.debugFailure(obj)
                raise AttributeError(f"Cannot pickle paths {failingPaths} of {obj}: {str(e)}")
        elif backend == "joblib":
            joblib.dump(obj, f, protocol=protocol)
        else:
            raise ValueError(f"Unknown backend '{backend}'")


class PickleFailureDebugger:
    """
    A collection of methods for testing whether objects can be pickled and logging useful infos in case they cannot
    """

    enabled = False  # global flag controlling the behaviour of logFailureIfEnabled

    @classmethod
    def _debugFailure(cls, obj, path, failures, handledObjectIds):
        if id(obj) in handledObjectIds:
            return
        handledObjectIds.add(id(obj))

        try:
            pickle.dumps(obj)
        except:
            # determine dictionary of children to investigate (if any)
            if hasattr(obj, '__dict__'):  # Because of strange behaviour of getstate, here try-except is used instead of if-else
                try:  # Because of strange behaviour of getattr(_, '__getstate__'), we here use try-except
                    d = obj.__getstate__()
                    if type(d) != dict:
                        d = {"state": d}
                except:
                    d = obj.__dict__
            elif type(obj) == dict:
                d = obj
            elif type(obj) in (list, tuple, set):
                d = dict(enumerate(obj))
            else:
                d = {}

            # recursively test children
            haveFailedChild = False
            for key, child in d.items():
                childPath = list(path) + [f"{key}[{child.__class__.__name__}]"]
                haveFailedChild = cls._debugFailure(child, childPath, failures, handledObjectIds) or haveFailedChild

            if not haveFailedChild:
                failures.append(path)

            return True
        else:
            return False

    @classmethod
    def debugFailure(cls, obj) -> List[str]:
        """
        Recursively tries to pickle the given object and returns a list of failed paths

        :param obj: the object for which to recursively test pickling
        :return: a list of object paths that failed to pickle
        """
        handledObjectIds = set()
        failures = []
        cls._debugFailure(obj, [obj.__class__.__name__], failures, handledObjectIds)
        return [".".join(l) for l in failures]

    @classmethod
    def logFailureIfEnabled(cls, obj, contextInfo: str = None):
        """
        If the class flag 'enabled' is set to true, the pickling of the given object is
        recursively tested and the results are logged at error level if there are problems and
        info level otherwise.
        If the flag is disabled, no action is taken.

        :param obj: the object for which to recursively test pickling
        :param contextInfo: optional additional string to be included in the log message
        """
        if cls.enabled:
            failures = cls.debugFailure(obj)
            prefix = f"Picklability analysis for {obj}"
            if contextInfo is not None:
                prefix += " (context: %s)" % contextInfo
            if len(failures) > 0:
                log.error(f"{prefix}: pickling would result in failures due to: {failures}")
            else:
                log.info(f"{prefix}: is picklable")


def setstate(cls, obj, state: Dict[str, Any], renamedProperties: Dict[str, str] = None, newOptionalProperties: List[str] = None,
        newDefaultProperties: Dict[str, Any] = None, removedProperties: List[str] = None) -> None:
    """
    Helper function for safe implementations of __setstate__ in classes, which appropriately handles the cases where
    a parent class already implements __setstate__ and where it does not. Call this function whenever you would actually
    like to call the super-class' implementation.
    Unfortunately, __setstate__ is not implemented in object, rendering super().__setstate__(state) invalid in the general case.

    :param cls: the class in which you are implementing __setstate__
    :param obj: the instance of cls
    :param state: the state dictionary
    :param renamedProperties: a mapping from old property names to new property names
    :param newOptionalProperties: a list of names of new property names, which, if not present, shall be initialised with None
    :param newDefaultProperties: a dictionary mapping property names to their default values, which shall be added if they are not present
    :param removedProperties: a list of names of properties that are no longer being used
    """
    # handle new/changed properties
    if renamedProperties is not None:
        for mOld, mNew in renamedProperties.items():
            if mOld in state:
                state[mNew] = state[mOld]
                del state[mOld]
    if newOptionalProperties is not None:
        for mNew in newOptionalProperties:
            if mNew not in state:
                state[mNew] = None
    if newDefaultProperties is not None:
        for mNew, mValue in newDefaultProperties.items():
            if mNew not in state:
                state[mNew] = mValue
    if removedProperties is not None:
        for p in removedProperties:
            if p in state:
                del state[p]
    # call super implementation, if any
    s = super(cls, obj)
    if hasattr(s, '__setstate__'):
        s.__setstate__(state)
    else:
        obj.__dict__ = state


def getstate(cls, obj, transientProperties: Iterable[str] = None, excludedProperties: Iterable[str] = None,
             overrideProperties: Dict[str, Any] = None, excludedDefaultProperties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Helper function for safe implementations of __getstate__ in classes, which appropriately handles the cases where
    a parent class already implements __getstate__ and where it does not. Call this function whenever you would actually
    like to call the super-class' implementation.
    Unfortunately, __getstate__ is not implemented in object, rendering super().__getstate__() invalid in the general case.

    :param cls: the class in which you are implementing __getstate__
    :param obj: the instance of cls
    :param transientProperties: transient properties which be set to None in serialisations
    :param excludedProperties: properties which shall be completely removed from serialisations
    :param overrideProperties: a mapping from property names to values specifying (new or existing) properties which are to be set;
        use this to set a fixed value for an existing property or to add a completely new property
    :param excludedDefaultProperties: properties which shall be completely removed from serialisations, if they are set
        to the given default value
    :return: the state dictionary, which may be modified by the receiver
    """
    s = super(cls, obj)
    if hasattr(s, '__getstate__'):
        d = s.__getstate__()
    else:
        d = obj.__dict__.copy()
    if transientProperties is not None:
        for p in transientProperties:
            if p in d:
                d[p] = None
    if excludedProperties is not None:
        for p in excludedProperties:
            if p in d:
                del d[p]
    if overrideProperties is not None:
        for k, v in overrideProperties.items():
            d[k] = v
    if excludedDefaultProperties is not None:
        for p, v in excludedDefaultProperties.items():
            if p in d and d[p] == v:
                del d[p]
    return d