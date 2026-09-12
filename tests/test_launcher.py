"""The ``sma-gui`` launcher degrades gracefully without the 'gui' extra.

The entry point is installed by every install, but PySide6/pyqtgraph ship only
in the optional extra. When they're absent the command must fail with an
actionable message and a non-zero exit code, not a raw ImportError traceback.
"""

from __future__ import annotations

import importlib.util
import sys


def test_launcher_reports_missing_gui(monkeypatch, capsys):
    # Simulate a base install: PySide6 unimportable, and none of the GUI package
    # already loaded. Purging the whole ``sound_metric_app.ui.*`` subtree, not just
    # ``main_window``, is what makes this independent of test ordering: main_window
    # is a package façade now, so a cached submodule left behind by an earlier test
    # would satisfy its imports from sys.modules and the ImportError would never
    # fire. ``ui.controller`` and ``ui.launcher`` go too — they are re-imported
    # from source below.
    monkeypatch.setitem(sys.modules, "PySide6", None)
    for name in [m for m in sys.modules if m.startswith("sound_metric_app.ui")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    from sound_metric_app.ui.launcher import main

    assert main() == 1
    assert "pip install sound-metric-app[gui]" in capsys.readouterr().err


def test_launcher_reports_missing_pyqtgraph(monkeypatch, capsys):
    # A partial 'gui' extra: PySide6 present (so ``main_window`` imports cleanly)
    # but pyqtgraph absent. pyqtgraph is imported lazily inside
    # ``MainWindow.__init__``, past the launcher's import guard, so the launcher
    # must probe for it separately rather than let a raw ImportError escape when
    # the window is built. ``_main`` must never run — patch it to prove it doesn't.
    from sound_metric_app.ui import launcher, main_window

    def _fail() -> int:  # pragma: no cover - only reached on regression
        raise AssertionError("launcher ran the app despite missing pyqtgraph")

    # launcher.main() does ``from .main_window import main as _main``; patch that
    # target so a regression that ran the app would blow up instead of passing.
    # Patch the imported module object (not a dotted string) so this does not
    # depend on ``main_window`` being cached as an attribute of the ``ui`` package,
    # which an earlier test may have purged.
    monkeypatch.setattr(main_window, "main", _fail, raising=False)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "pyqtgraph":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert launcher.main() == 1
    assert "pip install sound-metric-app[gui]" in capsys.readouterr().err
