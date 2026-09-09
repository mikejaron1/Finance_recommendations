"""Compatibility façade for the shared UI; implementation lives in :mod:`ui`."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from finrec import db, storage
from finrec.profile import Profile
if __package__:
    from .ui import theme, formatting, charts, state, widgets, chrome
    from .ui.rendering import st
else:
    from ui import theme, formatting, charts, state, widgets, chrome
    from ui.rendering import st

# Retain private helper imports used by older views and tests as well.
for _module in (theme, formatting, charts, state, widgets, chrome):
    globals().update({name: value for name, value in vars(_module).items()
                      if not name.startswith("__")})
