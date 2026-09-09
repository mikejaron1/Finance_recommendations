"""Explicit currency-safe Streamlit adapter, without global monkeypatches."""
from __future__ import annotations

import functools
import streamlit
from streamlit.delta_generator import DeltaGenerator

from .formatting import escape_dollars


class StreamlitUI:
    """Proxy text rendering and returned containers; leave values untouched."""

    _text_methods = frozenset({
        "markdown", "caption", "info", "success", "warning", "error",
        "expander", "write", "toast", "popover",
    })
    _container_methods = frozenset({
        "columns", "container", "empty", "tabs", "form", "status", "chat_message",
    })

    def __init__(self, target):
        self._target = target

    def __enter__(self):
        self._target.__enter__()
        return self

    def __exit__(self, *args):
        return self._target.__exit__(*args)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, DeltaGenerator):
            return cls(value)
        if isinstance(value, (list, tuple)):
            return type(value)(cls._wrap(item) for item in value)
        return value

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if isinstance(attr, DeltaGenerator):
            return self._wrap(attr)
        if not callable(attr) or name not in self._text_methods | self._container_methods:
            return attr

        @functools.wraps(attr)
        def call(*args, **kwargs):
            if name in self._text_methods:
                args = tuple(escape_dollars(arg) if isinstance(arg, str) else arg
                             for arg in args)
                for field in ("body", "label", "text"):
                    if isinstance(kwargs.get(field), str):
                        kwargs[field] = escape_dollars(kwargs[field])
            return self._wrap(attr(*args, **kwargs))

        call._finrec_dollar_safe = name in self._text_methods
        return call


st = StreamlitUI(streamlit)
