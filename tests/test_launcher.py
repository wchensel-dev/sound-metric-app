"""The ``sma-gui`` launcher degrades gracefully without the 'gui' extra.

The entry point is installed by every install, but PySide6/pyqtgraph ship only
in the optional extra. When they're absent the command must fail with an
actionable message and a non-zero exit code, not a raw ImportError traceback.
"""

from __future__ import annotations

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
