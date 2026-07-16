"""Console-friendly entry point for the ``sma-gui`` command.

The desktop app lives in :mod:`sound_metric_app.ui.main_window`, which imports
PySide6 at module load time. Those dependencies ship only in the optional
``gui`` extra, but the ``sma-gui`` script is installed by every install. This
launcher defers the heavy import so a base install (``pip install
sound-metric-app`` without ``[gui]``) fails with an actionable message instead
of a raw :class:`ImportError` traceback.
"""

from __future__ import annotations

import sys

_INSTALL_HINT = (
    "The 'sma-gui' desktop app requires the optional GUI dependencies "
    "(PySide6, pyqtgraph).\n"
    "Install them with:  pip install sound-metric-app[gui]"
)


def main() -> int:
    try:
        from .main_window import main as _main
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 1
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
