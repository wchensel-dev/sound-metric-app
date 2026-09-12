"""PySide6 desktop app for the ingest -> mark -> bring-forward -> report workflow.

The window itself: five tabs, a Settings menu, and the coordination between them.
The tabs live in :mod:`sound_metric_app.ui.views` (which documents what each one
is for); the graph they share is :mod:`sound_metric_app.ui.graph`.

What stays here is what only the window can own — creating the five views over
one :class:`~sound_metric_app.ui.controller.WorkflowController`, and the three
cross-tab actions no single tab can perform:

* :meth:`MainWindow.notify_changed` — the post-mutation reload. Ingest, mark,
  include and close all end here, and it is also where the destructive
  empty-container sweep and the Compare tab's cache drop are tied to a mutation
  rather than to navigation.
* :meth:`MainWindow.open_marking_for` — the Ingest tab handing a shot to Mark.
* :meth:`MainWindow.add_to_compare` / :meth:`MainWindow.update_compare_count` —
  the Data bank and Batch average tabs pinning a curve to Compare without
  leaving the tab they are on.

Run with:  python -m sound_metric_app.ui.main_window   (needs the 'gui' extra)
"""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .compare_series import CompareSeries
from .controller import WorkflowController
from .dialogs import AmmoDefinitionsDialog
from .views import (
    BatchAverageView,
    CompareView,
    DataBankView,
    IngestView,
    MarkingView,
)

#: The Compare tab's title, which grows a count of what is pinned to it (see
#: :meth:`MainWindow.update_compare_count`).
_COMPARE_TAB_LABEL = "Compare"


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: WorkflowController | None = None):
        super().__init__()
        self.setWindowTitle("Sound Metric App — Workflow")
        # Wider than the other tabs need: the Report tab splits into a tree on the
        # left and a metric graph on the right, so give both room by default.
        self.resize(1100, 620)
        self.controller = controller or WorkflowController()

        self.ingest_view = IngestView(self.controller, self)
        self.marking_view = MarkingView(self.controller, self)
        self.bank_view = DataBankView(self.controller, self)
        self.report_view = BatchAverageView(self.controller, self)
        self.compare_view = CompareView(self.controller, self)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.ingest_view, "Ingest")
        self.tabs.addTab(self.marking_view, "Mark")
        self.tabs.addTab(self.bank_view, "Data bank")
        self.tabs.addTab(self.report_view, "Batch average")
        self.tabs.addTab(self.compare_view, _COMPARE_TAB_LABEL)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self._views = [
            self.ingest_view,
            self.marking_view,
            self.bank_view,
            self.report_view,
            self.compare_view,
        ]
        self._build_menus()
        self.notify_changed()

    def _build_menus(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")
        ammo_action = settings_menu.addAction("Ammo definitions…")
        ammo_action.triggered.connect(self._edit_ammo_definitions)

    def _edit_ammo_definitions(self) -> None:
        """Open the ammo-preset editor; on save, persist and refresh the mark form."""
        dialog = AmmoDefinitionsDialog(self.controller.ammo_definitions(), parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self.controller.set_ammo_definitions(dialog.definitions())
        self.notify_changed()

    def _on_tab_changed(self, index: int) -> None:
        self.tabs.widget(index).refresh()

    def notify_changed(self) -> None:
        """Reload every view after a mutating action (ingest / mark / include / close)."""
        # A mutation is the only thing that orphans a cluster/batch/combination (a
        # re-mark or edit can empty the old container), so the destructive sweep is
        # tied to the mutation here rather than to every refresh() — pure navigation
        # must never delete rows.
        self.controller.sweep_empty()
        # Same reasoning for the Compare tab's memoized curves: a mutation can
        # change what a pinned curve is drawn *from* (a re-mark retags which
        # channel is ML), so they are dropped here and not on every refresh.
        self.compare_view.invalidate_traces()
        for view in self._views:
            view.refresh()

    def open_marking_for(self, shot_id: int) -> None:
        """Switch to the Mark tab focused on ``shot_id`` (from the Ingest view)."""
        self.marking_view.select_shot(shot_id)
        self.tabs.setCurrentWidget(self.marking_view)

    def add_to_compare(self, series: CompareSeries) -> bool:
        """Pin one shot/mic to the Compare tab. False if it was already there.

        Deliberately does *not* switch tabs: pinning is done from the Batch
        average tree, several rows at a time (see
        :meth:`~sound_metric_app.ui.views.batch_average.BatchAverageView._pin_for_compare`).
        """
        return self.compare_view.add_series(series)

    def update_compare_count(self, count: int) -> None:
        """Carry how many shots are pinned on the Compare tab's own label.

        Pinning happens on another tab and draws nothing there, so the count is
        what tells the operator the click landed — and what the overlay is
        currently worth switching to.
        """
        index = self.tabs.indexOf(self.compare_view)
        self.tabs.setTabText(
            index, f"{_COMPARE_TAB_LABEL} ({count})" if count else _COMPARE_TAB_LABEL
        )


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
