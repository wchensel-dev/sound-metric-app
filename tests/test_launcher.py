"""The ``sma-gui`` launcher degrades gracefully without the 'gui' extra.

The entry point is installed by every install, but PySide6/pyqtgraph ship only
in the optional extra. When they're absent the command must fail with an
actionable message and a non-zero exit code, not a raw ImportError traceback.
"""

from __future__ import annotations

import sys


def test_launcher_reports_missing_gui(monkeypatch, capsys):
    # Simulate a base install: PySide6 unimportable, main_window not yet loaded.
    monkeypatch.setitem(sys.modules, "PySide6", None)
    monkeypatch.delitem(sys.modules, "sound_metric_app.ui.main_window", raising=False)

    from sound_metric_app.ui.launcher import main

    assert main() == 1
    assert "pip install sound-metric-app[gui]" in capsys.readouterr().err
