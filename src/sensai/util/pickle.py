import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Iterable, Union

import joblib

from .io import is_s3_path, S3Object

log = logging.getLogger(__name__)


def load_pickle(path: Union[str, Path], backend="pickle"):
    if isinstance(path, Path):
        path = str(path)

    def read_file(f):
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

    if is_s3_path(path):
        return read_file(S3Object(path).open_file("rb"))
    with open(path, "rb") as f:
        return read_file(f)


def dump_pickle(obj, pickle_path: Union[str, Path], backend="pickle", protocol=pickle.HIGHEST_PROTOCOL):
    if isinstance(pickle_path, Path):
        pickle_path = str(pickle_path)

    def open_file():
        if is_s3_path(pickle_path):
            return S3Object(pickle_path).open_file("wb")
        else:
            return open(pickle_path, "wb")

    dir_name = os.path.dirname(pickle_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    with open_file() as f:
        if backend == "pickle":
            try:
                pickle.dump(obj, f, protocol=protocol)
            except AttributeError as e:
                failing_paths = PickleFailureDebugger.debug_failure(obj)
                raise AttributeError(f"Cannot pickle paths {failing_paths} of {obj}: {str(e)}")
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
    def _debug_failure(cls, obj, path, failures, handled_object_ids):
        if id(obj) in handled_object_ids:
            return
        handled_object_ids.add(id(obj))

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
            have_failed_child = False
            for key, child in d.items():
                child_path = list(path) + [f"{key}[{child.__class__.__name__}]"]
                have_failed_child = cls._debug_failure(child, child_path, failures, handled_object_ids) or have_failed_child

            if not have_failed_child:
                failures.append(path)

            return True
        else:
            return False

    @classmethod
    def debug_failure(cls, obj) -> List[str]:
        """
        Recursively tries to pickle the given object and returns a list of failed paths

        :param obj: the object for which to recursively test pickling
        :return: a list of object paths that failed to pickle
        """
        handled_object_ids = set()
        failures = []
        cls._debug_failure(obj, [obj.__class__.__name__], failures, handled_object_ids)
        return [".".join(l) for l in failures]

    @classmethod
    def log_failure_if_enabled(cls, obj, context_info: str = None):
        """
        If the class flag 'enabled' is set to true, the pickling of the given object is
        recursively tested and the results are logged at error level if there are problems and
        info level otherwise.
        If the flag is disabled, no action is taken.

        :param obj: the object for which to recursively test pickling
        :param context_info: optional additional string to be included in the log message
        """
        if cls.enabled:
            failures = cls.debug_failure(obj)
            prefix = f"Picklability analysis for {obj}"
            if context_info is not None:
                prefix += " (context: %s)" % context_info
            if len(failures) > 0:
                log.error(f"{prefix}: pickling would result in failures due to: {failures}")
            else:
                log.info(f"{prefix}: is picklable")


def setstate(cls,
        obj,
        state: Dict[str, Any],
        renamed_properties: Dict[str, str] = None,
        new_optional_properties: List[str] = None,
        new_default_properties: Dict[str, Any] = None,
        removed_properties: List[str] = None) -> None:
    """
    Helper function for safe implementations of __setstate__ in classes, which appropriately handles the cases where
    a parent class already implements __setstate__ and where it does not. Call this function whenever you would actually
    like to call the super-class' implementation.
    Unfortunately, __setstate__ is not implemented in object, rendering super().__setstate__(state) invalid in the general case.

    :param cls: the class in which you are implementing __setstate__
    :param obj: the instance of cls
    :param state: the state dictionary
    :param renamed_properties: a mapping from old property names to new property names
    :param new_optional_properties: a list of names of new property names, which, if not present, shall be initialised with None
    :param new_default_properties: a dictionary mapping property names to their default values, which shall be added if they are not present
    :param removed_properties: a list of names of properties that are no longer being used
    """
    # handle new/changed properties
    if renamed_properties is not None:
        for mOld, mNew in renamed_properties.items():
            if mOld in state:
                state[mNew] = state[mOld]
                del state[mOld]
    if new_optional_properties is not None:
        for mNew in new_optional_properties:
            if mNew not in state:
                state[mNew] = None
    if new_default_properties is not None:
        for mNew, mValue in new_default_properties.items():
            if mNew not in state:
                state[mNew] = mValue
    if removed_properties is not None:
        for p in removed_properties:
            if p in state:
                del state[p]
    # call super implementation, if any
    s = super(cls, obj)
    if hasattr(s, '__setstate__'):
        s.__setstate__(state)
    else:
        obj.__dict__ = state


def getstate(cls,
        obj,
        transient_properties: Iterable[str] = None,
        excluded_properties: Iterable[str] = None,
        override_properties: Dict[str, Any] = None,
        excluded_default_properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Helper function for safe implementations of __getstate__ in classes, which appropriately handles the cases where
    a parent class already implements __getstate__ and where it does not. Call this function whenever you would actually
    like to call the super-class' implementation.
    Unfortunately, __getstate__ is not implemented in object, rendering super().__getstate__() invalid in the general case.

    :param cls: the class in which you are implementing __getstate__
    :param obj: the instance of cls
    :param transient_properties: transient properties which be set to None in serialisations
    :param excluded_properties: properties which shall be completely removed from serialisations
    :param override_properties: a mapping from property names to values specifying (new or existing) properties which are to be set;
        use this to set a fixed value for an existing property or to add a completely new property
    :param excluded_default_properties: properties which shall be completely removed from serialisations, if they are set
        to the given default value
    :return: the state dictionary, which may be modified by the receiver
    """
    s = super(cls, obj)
    if hasattr(s, '__getstate__'):
        d = s.__getstate__()
    else:
        d = obj.__dict__.copy()
    if transient_properties is not None:
        for p in transient_properties:
            if p in d:
                d[p] = None
    if excluded_properties is not None:
        for p in excluded_properties:
            if p in d:
                del d[p]
    if override_properties is not None:
        for k, v in override_properties.items():
            d[k] = v
    if excluded_default_properties is not None:
        for p, v in excluded_default_properties.items():
            if p in d and d[p] == v:
                del d[p]
    return d
