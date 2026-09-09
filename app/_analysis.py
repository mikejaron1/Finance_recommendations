"""Bounded, session-local memoization of pure analyses, never hosted requests."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections import OrderedDict

import numpy as np
import streamlit as st


def _json_default(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot cache an input of type {type(value).__name__}.")


def cached_call(function, *args, **kwargs):
    """Cache by all inputs in this browser session; return an independent result."""
    serialized = json.dumps(
        [function.__module__, function.__qualname__, args, kwargs],
        sort_keys=True, default=_json_default, allow_nan=False,
    )
    key = hashlib.sha256(serialized.encode()).hexdigest()
    if "_analysis_results" not in st.session_state:
        st.session_state["_analysis_results"] = OrderedDict()
    cache = st.session_state["_analysis_results"]
    if key not in cache:
        cache[key] = function(*args, **kwargs)
        while len(cache) > 8:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return copy.deepcopy(cache[key])
