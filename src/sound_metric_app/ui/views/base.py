"""The base every workflow tab is built on, plus the worker thread it runs on.

Two things live here. :class:`_Task` is the off-thread runner: the two
file-reading operations (ingest, mark) and every capture read behind a graph go
through it, so a large capture never freezes the window. :class:`_View` is what
the five tabs subclass — it holds the controller and the coordinating
:class:`~sound_metric_app.ui.main_window.MainWindow`, and carries the handful of
behaviours every tab needs: run something off the UI thread, defer a callback
past the signal currently being emitted, and build the shared SKU filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from ..controller import WorkflowController

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from ..main_window import MainWindow


class _Task(QtCore.QThread):
    """Run a no-arg callable on a worker thread; emit its result or exception.

    The controller opens its own SQLite connection per call, so running one of
    its methods here is thread-safe: nothing touches a connection owned by the
    UI thread. Widgets are never touched from ``run``; results come back via the
    queued-connection signals.
    """

    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # executed on the worker thread
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 — reported to the UI as a dialog
            self.failed.emit(exc)
        else:
            self.succeeded.emit(result)


class _View(QtWidgets.QWidget):
    """Base view holding the controller, coordinator, and the async helper."""

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__()
        self.controller = controller
        self.main = main
        self._tasks: set[_Task] = set()

    def refresh(self) -> None:  # overridden by views that show live data
        """Reload this view's data from the controller."""

    def _add_sku_filter(self, row: QtWidgets.QHBoxLayout) -> QtWidgets.QComboBox:
        """Build the shared SKU-filter dropdown into ``row`` and return it.

        Both archive views (data bank, batch average) front their tree with the
        same filter: a ``SKU:`` label and a combo whose ``currentIndexChanged``
        re-runs this view's ``refresh``. The rows are (re)filled at refresh time
        by :func:`~sound_metric_app.ui.qtwidgets._repopulate_sku_filter`, and the
        caller reads the choice back with ``currentData()``. Kept here so the
        wiring lives in one place and the two views can't drift.
        """
        combo = QtWidgets.QComboBox()
        combo.currentIndexChanged.connect(self.refresh)
        row.addWidget(QtWidgets.QLabel("SKU:"))
        row.addWidget(combo)
        return combo

    def _prompt_discard_reason(self, title: str, message: str, default: str = "") -> str | None:
        """Ask for an optional multi-line reason before a discard; ``None`` on Cancel.

        Shared by the Ingest table, the Ingest bad-files row, and the Marking
        tab so their discard-confirmation copy and validation can't drift
        apart between the three.
        """
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self, title, message, default)
        if not ok:
            return None
        return text.strip()

    def _defer(self, fn) -> None:
        """Run ``fn`` from the event loop once the current signal has unwound.

        The escape hatch for a handler that rebuilds the very widget whose
        signal invoked it: a refresh clears its tree, and freeing the row Qt is
        still emitting for is a use-after-free that takes the app down. Passing
        ``self`` as the context object drops the call if this view is destroyed
        before it fires.
        """
        QtCore.QTimer.singleShot(0, self, fn)

    def _run_async(self, fn, on_success, *, busy=()) -> None:
        """Run ``fn`` off the UI thread; call ``on_success(result)`` when done.

        ``busy`` widgets are disabled and a wait cursor shown for the duration.
        Any exception becomes a critical dialog instead of a crash.
        """
        for w in busy:
            w.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        task = _Task(fn, self)
        self._tasks.add(task)

        def cleanup() -> None:
            QtWidgets.QApplication.restoreOverrideCursor()
            for w in busy:
                w.setEnabled(True)
            self._tasks.discard(task)

        def handle_success(result) -> None:
            cleanup()
            on_success(result)

        def handle_failure(exc) -> None:
            cleanup()
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))

        task.succeeded.connect(handle_success)
        task.failed.connect(handle_failure)
        task.finished.connect(task.deleteLater)
        task.start()
