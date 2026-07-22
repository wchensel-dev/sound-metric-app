"""Qt smoke/acceptance test for the workflow window.

Drives ingest -> mark -> close -> report through the real widgets (buttons,
combos, tables), so the off-thread task wiring and cross-view refresh are
exercised, not just the controller. Skipped when the ``gui`` extra is absent.

Run headless:  QT_QPA_PLATFORM=offscreen pytest tests/test_ui.py
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore  # noqa: E402

from sound_metric_app.ingestion import ChannelInfo  # noqa: E402
from sound_metric_app.models import Frame  # noqa: E402
from sound_metric_app.ui.controller import WorkflowController  # noqa: E402
from sound_metric_app.ui.main_window import MainWindow  # noqa: E402

FS = 200_000.0


def _sine_frame(path: str, channel: str) -> Frame:
    t = np.arange(20_000) / FS
    return Frame(
        samples=np.sin(2 * np.pi * 1000.0 * t),
        sample_rate=FS,
        channel=channel,
        source_file=path,
        timestamp=None,
    )


def _fake_channels(path: str) -> list[ChannelInfo]:
    return [
        ChannelInfo(name="AI 1", unit="Pa", sample_rate=FS, n_samples=20_000),
        ChannelInfo(name="AI 2", unit="Pa", sample_rate=FS, n_samples=20_000),
    ]


def _fake_capture(path: str) -> list[Frame]:
    return [_sine_frame(path, "AI 1"), _sine_frame(path, "AI 2")]


@pytest.fixture
def window(tmp_path, monkeypatch, qtbot):
    monkeypatch.setenv("SMA_CONFIG", str(tmp_path / "sma_config.json"))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("SUP-1_AR15_001.dxd", "SUP-1_AR15_002.dxd"):
        (inbox / name).write_bytes(b"")

    controller = WorkflowController(
        tmp_path / "wf.db",
        channel_reader=_fake_channels,
        capture_reader=_fake_capture,
    )
    controller.set_input_folder(inbox)
    win = MainWindow(controller)
    qtbot.addWidget(win)
    return win


def test_format_metric_blanks_null_instead_of_raising():
    # channel_metrics columns are nullable REAL (the schema-v1 migration blanks
    # peak_impulse_db), so a metric can be None. It must render as an em-dash,
    # not raise TypeError from f"{None:.2f}" and abort the whole report render.
    from sound_metric_app.ui.main_window import _format_metric

    assert _format_metric(None) == "—"
    assert _format_metric(163.4) == "163.40"
    assert _format_metric(0) == "0.00"


def test_report_no_metrics_row_spans_all_columns(window, monkeypatch):
    # A group with no averages renders a "no metrics" placeholder row. It must
    # carry a cell for every column so it stays aligned with the widened header
    # (and keeps tracking _METRIC_KEYS if the metric set grows again), rather
    # than under-filling and leaving trailing columns without a cell.
    from sound_metric_app.models import Batch, Group
    from sound_metric_app.services.aggregation_service import BatchReport, GroupAverages

    rv = window.report_view
    report = BatchReport(
        batch=Batch(sku="SUP-1", id=1),
        groups=[
            GroupAverages(
                group=Group(test_platform="AR15", ammo="M855", id=1),
                n_shots=0,
                averages={},
                shots={},
            )
        ],
    )
    monkeypatch.setattr(rv.controller, "batch_report", lambda _batch_id: report)
    rv.batch_combo.blockSignals(True)
    rv.batch_combo.addItem("#1", 1)  # give _load_report a non-None batch id
    rv.batch_combo.blockSignals(False)

    rv._load_report()

    assert rv.tree.topLevelItemCount() == 1
    item = rv.tree.topLevelItem(0)
    # Every column has a cell -> nothing shifts left; the row matches the header.
    assert item.columnCount() == len(rv._COLUMNS)
    assert item.text(0) == "AR15 / M855"
    assert item.text(rv._FIRST_METRIC_COL) == "no metrics"


def test_window_builds_with_four_tabs(window):
    assert window.tabs.count() == 4
    assert [window.tabs.tabText(i) for i in range(4)] == ["Ingest", "Mark", "Batches", "Report"]


def test_selecting_shot_with_null_keys_does_not_crash_mark_tab(window):
    # A shot whose filename yielded no batch/group keys is stored with
    # suppressor_sku/test_platform = None. Selecting it must not pass None to
    # QLineEdit.setPlaceholderText (which raises TypeError).
    with window.controller._repo() as repo:
        shot_id = repo.add_unmarked_shot("no-keys.dxd", None, None, 1)

    mv = window.marking_view
    mv.refresh()  # _on_shot_changed fires on selection; must not raise
    mv.shot_combo.setCurrentIndex(mv._index_of_shot(shot_id))

    assert mv._current_shot_id() == shot_id
    assert mv.sku_edit.placeholderText() == ""
    assert mv.platform_edit.placeholderText() == ""


def _mark_all_shots(window, qtbot):
    """Ingest the fixture inbox and mark every shot (SE=AI 1, MR=AI 2)."""
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    while window.ingest_view.table.rowCount() > 0:
        first_id = int(window.ingest_view.table.item(0, 0).text())
        window.open_marking_for(first_id)
        mv = window.marking_view
        qtbot.waitUntil(
            lambda: mv.se_combo.isEnabled() and mv.se_combo.count() >= 3, timeout=5000
        )
        mv.ammo_combo.setCurrentText("M855")
        before = window.ingest_view.table.rowCount()
        mv._mark()
        qtbot.waitUntil(
            lambda: window.ingest_view.table.rowCount() < before, timeout=5000
        )


def test_clicking_metric_cell_graphs_that_shot(window, qtbot):
    from sound_metric_app.models import MicPosition

    _mark_all_shots(window, qtbot)

    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() >= 2, timeout=5000)

    # Drill into an SE average row and grab one of its shot children.
    se_top = next(
        rv.tree.topLevelItem(i)
        for i in range(rv.tree.topLevelItemCount())
        if rv.tree.topLevelItem(i).text(1) == "SE"
    )
    shot_item = se_top.child(0)
    kind, _shot_id, position = shot_item.data(0, QtCore.Qt.UserRole)
    assert kind == "shot" and position == MicPosition.SE

    # Click the first metric cell (Peak Pa) -> one curve is drawn.
    peak_col = rv._FIRST_METRIC_COL
    rv._on_cell_clicked(shot_item, peak_col)
    qtbot.waitUntil(lambda: len(rv.graph._plot.listDataItems()) >= 1, timeout=5000)

    # Auto Frame becomes usable once a trace is drawn, and snaps X to the shot
    # window's full width (first/last sample), which the yellow bounds mark.
    assert rv.graph._auto_frame_btn.isEnabled()
    assert rv.graph._x_bounds is not None
    x0, x1 = rv.graph._x_bounds
    rv.graph.auto_frame()
    view_x0, view_x1 = rv.graph._plot.getViewBox().viewRange()[0]
    assert view_x0 == pytest.approx(x0)
    assert view_x1 == pytest.approx(x1)

    # Clicking a non-metric column just prompts; it doesn't graph, and Auto Frame
    # goes back to disabled with no bounds.
    rv._on_cell_clicked(shot_item, 0)
    assert len(rv.graph._plot.listDataItems()) == 0
    assert not rv.graph._auto_frame_btn.isEnabled()
    assert rv.graph._x_bounds is None


def test_auto_frame_bounds_track_finite_curve_extent(qtbot):
    # A NaN-padded curve (the Impulse ∫p·dt trace is NaN before the onset)
    # must frame to where the curve actually exists, not the full sample axis --
    # otherwise Auto Frame stretches X across a sea of empty samples.
    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.main_window import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)

    trace = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        values=np.array([np.nan, 10.0, 20.0, 15.0, np.nan]),
        y_label="Impulse ∫p·dt (Pa·ms)",
        title="Peak Impulse",
        connected=True,
    )
    graph.show_trace(trace)
    # Bounds bracket the finite span [1.0, 3.0], not the raw axis [0.0, 4.0].
    assert graph._x_bounds == (1.0, 3.0)
    graph.auto_frame()
    view_x0, view_x1 = graph._plot.getViewBox().viewRange()[0]
    assert view_x0 == pytest.approx(1.0)
    assert view_x1 == pytest.approx(3.0)


def test_graph_draws_calculation_window_markers(qtbot):
    # The dashed verticals that separate "samples the metric used" from the
    # context drawn either side. Also exercises the InfiniteLine label options.
    import pyqtgraph as pg

    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.main_window import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)

    def verticals():
        return [
            item.value()
            for item in graph._plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine) and item.angle == 90
        ]

    trace = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        values=np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        y_label="SPL (dB)",
        title="Peak dB",
        peak_index=2,
        window_start_index=1,
        window_end_index=3,
    )
    graph.show_trace(trace)
    drawn = verticals()
    assert 1.0 in drawn, "no vertical at the window start"
    assert 3.0 in drawn, "no vertical at the window end"

    # Each is independent: an edge outside the capture simply isn't drawn.
    trace.window_end_index = None
    graph.show_trace(trace)
    drawn = verticals()
    assert 1.0 in drawn
    assert 3.0 not in drawn


def test_frame_calc_window_zooms_to_the_window_edges(qtbot):
    # Frame Calc Window is Auto Frame's zoomed-in counterpart: it snaps X to the
    # two dashed window lines rather than the curve's full extent, and needs both
    # edges to have something to frame.
    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.main_window import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)

    trace = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        values=np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        y_label="SPL (dB)",
        title="Peak dB",
        window_start_index=1,
        window_end_index=3,
    )
    graph.show_trace(trace)
    assert graph._frame_window_btn.isEnabled()
    assert graph._window_x_bounds == (1.0, 3.0)
    graph.frame_calc_window()
    view_x0, view_x1 = graph._plot.getViewBox().viewRange()[0]
    # Framed to the window, not the curve's [0.0, 4.0] extent; the small padding
    # keeps both lines off the very edge.
    assert 0.0 < view_x0 < 1.0
    assert 3.0 < view_x1 < 4.0

    # A trace with only one window edge -- or none at all -- has no span to
    # frame, so the button goes back to disabled.
    trace.window_end_index = None
    graph.show_trace(trace)
    assert not graph._frame_window_btn.isEnabled()
    assert graph._window_x_bounds is None

    graph.show_message("nothing graphed")
    assert not graph._frame_window_btn.isEnabled()


def test_window_marker_labels_run_vertically_from_the_top(qtbot):
    # The labels are rotated parallel to their line and top-aligned. Guards the
    # anchor choice: pyqtgraph's default anchors for rotated text centre the
    # label on `position`, which with position~1.0 hangs half of it above the
    # view and clips it. Anchoring the far end pins the top edge instead.
    import pyqtgraph as pg

    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.main_window import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)
    graph.resize(900, 500)
    graph.show()

    trace = MetricTrace(
        t_ms=np.linspace(0.0, 210.0, 2100),
        values=np.linspace(100.0, 150.0, 2100),
        y_label="SPL (dB)",
        title="Peak dB",
        window_start_index=100,
        window_end_index=1100,
    )
    graph.show_trace(trace)
    graph.grab()  # force a paint pass so pyqtgraph applies the label transform

    view_rect = graph._plot.getPlotItem().vb.sceneBoundingRect()
    labels = [
        child
        for item in graph._plot.getPlotItem().items
        if isinstance(item, pg.InfiniteLine)
        for child in item.childItems()
        if isinstance(child, pg.InfLineLabel)
    ]
    assert len(labels) == 2

    for label in labels:
        rect = label.mapRectToScene(label.boundingRect())
        assert rect.height() > rect.width(), f"{label.format} is not rotated"
        assert rect.top() >= view_rect.top(), f"{label.format} clipped above the view"
        # Top-aligned: snug under the top edge, not floating mid-plot.
        assert rect.top() - view_rect.top() < view_rect.height() * 0.1


def test_graph_point_readout_shows_value_and_clears(qtbot):
    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.main_window import MetricGraph, _unit_of

    assert _unit_of("SPL (dBA)") == "dBA"
    assert _unit_of("Pressure (Pa)") == "Pa"
    assert _unit_of("no parens") == "no parens"

    graph = MetricGraph()
    qtbot.addWidget(graph)

    trace = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0]),
        values=np.array([100.0, 142.5, np.nan]),
        y_label="SPL (dBA)",
        title="Peak dBA",
        peak_index=1,
    )
    graph.show_trace(trace)
    # Nothing picked yet: the readout box is hidden.
    assert not graph._readout_label.isVisible()
    assert graph._pick_marker is None

    # Picking a sample fills the box with its value + unit + time and marks it.
    graph._show_readout(1, 142.5)
    assert graph._pick_marker is not None
    text = graph._readout_label.text()
    assert "142.500" in text and "dBA" in text and "1.00 ms" in text

    # Clear removes the marker and hides the box.
    graph.clear_readout()
    assert graph._pick_marker is None
    assert not graph._readout_label.isVisible()

    # Drawing a fresh trace also drops any prior pick.
    graph._show_readout(0, 100.0)
    graph.show_trace(trace)
    assert graph._pick_marker is None
    assert not graph._readout_label.isVisible()


def test_full_workflow_through_widgets(window, qtbot):
    # --- Ingest (off-thread) -> two unmarked rows ---
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)

    # --- Mark both shots via the marking form ---
    for _ in range(2):
        first_id = int(window.ingest_view.table.item(0, 0).text())
        window.open_marking_for(first_id)
        mv = window.marking_view
        # Wait for the (fake) channel load to populate the SE picker.
        qtbot.waitUntil(lambda: mv.se_combo.isEnabled() and mv.se_combo.count() >= 3, timeout=5000)
        mv.ammo_combo.setCurrentText("M855")  # SE/MR default to AI 1 / AI 2
        mv._mark()
        qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() < 2, timeout=5000)
        # loop condition re-reads the table; wait for the second mark to clear it
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 0, timeout=5000)

    # --- Report shows the group with SE and MR rows (never mixed) ---
    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() >= 2, timeout=5000)
    tops = [rv.tree.topLevelItem(i) for i in range(rv.tree.topLevelItemCount())]
    mics = {item.text(1) for item in tops}
    assert {"SE", "MR"} <= mics
    # Each mic average expands to its individual shots.
    se_item = next(item for item in tops if item.text(1) == "SE")
    assert se_item.childCount() >= 1

    # --- Close the batch from the tree ---
    tree = window.batch_view.tree
    tree.setCurrentItem(tree.topLevelItem(0))
    assert window.batch_view.close_btn.isEnabled()


def _first_batch_item(bv):
    return bv.tree.topLevelItem(0)


def test_edit_button_enabled_for_batch_and_shot_not_group(window, qtbot):
    # Mark the two shots so the tree has a batch -> group -> shots to select in.
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    for _ in range(2):
        first_id = int(window.ingest_view.table.item(0, 0).text())
        window.open_marking_for(first_id)
        mv = window.marking_view
        qtbot.waitUntil(lambda: mv.se_combo.isEnabled() and mv.se_combo.count() >= 3, timeout=5000)
        mv.ammo_combo.setCurrentText("M855")
        mv._mark()
        qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() < 2, timeout=5000)
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 0, timeout=5000)

    bv = window.batch_view
    bv.refresh()
    batch_item = _first_batch_item(bv)
    group_item = batch_item.child(0)
    shot_item = group_item.child(0)

    bv.tree.setCurrentItem(batch_item)
    assert bv.edit_btn.isEnabled()  # batch: rename SKU
    bv.tree.setCurrentItem(group_item)
    assert not bv.edit_btn.isEnabled()  # group: no direct edit
    bv.tree.setCurrentItem(shot_item)
    assert bv.edit_btn.isEnabled()  # shot: re-mark


def test_rename_batch_via_tree(window, qtbot, monkeypatch):
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    first_id = int(window.ingest_view.table.item(0, 0).text())
    window.open_marking_for(first_id)
    mv = window.marking_view
    qtbot.waitUntil(lambda: mv.se_combo.isEnabled() and mv.se_combo.count() >= 3, timeout=5000)
    mv.ammo_combo.setCurrentText("M855")
    mv._mark()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 1, timeout=5000)

    bv = window.batch_view
    bv.refresh()
    bv.tree.setCurrentItem(_first_batch_item(bv))

    # Stand in for the modal SKU prompt with a fixed corrected value.
    from PySide6 import QtWidgets

    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", staticmethod(lambda *a, **k: ("SUP-FIXED", True))
    )
    bv._edit_selected()

    assert window.controller.batches()[0].sku == "SUP-FIXED"
    assert _first_batch_item(bv).text(0).endswith("SKU SUP-FIXED")


def test_double_click_edits_shots_not_parent_rows(window, qtbot, monkeypatch):
    # Double-click is Qt's expand/collapse gesture on batch/group rows; it must
    # not also route through _edit_selected there (only leaf shot rows edit).
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    first_id = int(window.ingest_view.table.item(0, 0).text())
    window.open_marking_for(first_id)
    mv = window.marking_view
    qtbot.waitUntil(lambda: mv.se_combo.isEnabled() and mv.se_combo.count() >= 3, timeout=5000)
    mv.ammo_combo.setCurrentText("M855")
    mv._mark()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 1, timeout=5000)

    bv = window.batch_view
    bv.refresh()
    edited: list = []
    monkeypatch.setattr(bv, "_edit_selected", lambda: edited.append(True))

    batch_item = _first_batch_item(bv)
    group_item = batch_item.child(0)
    shot_item = group_item.child(0)

    bv._on_item_double_clicked(batch_item, 0)
    bv._on_item_double_clicked(group_item, 0)
    assert edited == []  # parent rows: no edit modal

    bv._on_item_double_clicked(shot_item, 0)
    assert edited == [True]  # leaf shot row: edits


def test_edit_shot_re_marks_with_corrected_ammo(window, qtbot):
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    first_id = int(window.ingest_view.table.item(0, 0).text())
    window.open_marking_for(first_id)
    mv = window.marking_view
    qtbot.waitUntil(lambda: mv.se_combo.isEnabled() and mv.se_combo.count() >= 3, timeout=5000)
    mv.ammo_combo.setCurrentText("WRONG")
    mv._mark()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 1, timeout=5000)

    from PySide6 import QtCore

    bv = window.batch_view
    bv.refresh()
    shot_item = _first_batch_item(bv).child(0).child(0)
    _, shot, group, batch = shot_item.data(0, QtCore.Qt.UserRole)

    # Open the pre-filled dialog directly (bypassing the async channel load),
    # correct the ammo, and accept it as the user would.
    from sound_metric_app.ui.main_window import ShotEditDialog

    dialog = ShotEditDialog(
        shot,
        sku=batch.sku,
        platform=group.test_platform,
        ammo=group.ammo,
        channel_names=["AI 1", "AI 2"],
        parent=bv,
    )
    assert dialog.ammo_combo.currentText() == "WRONG"  # pre-filled from the group
    assert dialog.se_combo.currentText() == "AI 1"  # pre-filled from the shot tags
    dialog.ammo_combo.setCurrentText("M855")
    dialog._on_accept()

    bv._run_async(
        lambda: window.controller.mark(shot.id, **dialog.values()),
        lambda _r: window.notify_changed(),
    )
    # The shot moves to the corrected "M855" group; the emptied "WRONG" group is
    # dropped so the batch tree does not keep an empty group behind.
    qtbot.waitUntil(
        lambda: {g.ammo for g in window.controller.groups_for_batch(batch.id)} == {"M855"},
        timeout=5000,
    )
    by_ammo = {
        g.ammo: window.controller.shots_by_group(g.id)
        for g in window.controller.groups_for_batch(batch.id)
    }
    assert "WRONG" not in by_ammo
    assert [s.id for s in by_ammo["M855"]] == [shot.id]


def test_mark_tab_offers_configured_ammo_presets(window):
    # The mark form's ammo combo is seeded with the default presets and reflects
    # a saved custom list after Settings ▸ Ammo definitions.
    mv = window.marking_view
    presets = [mv.ammo_combo.itemText(i) for i in range(mv.ammo_combo.count())]
    assert presets == ["LC M193 (5.56)", "LC M855 (5.56)", "Black Hills 77gr OTM (5.56)"]

    window.controller.set_ammo_definitions(["Custom 62gr", "LC M855 (5.56)"])
    window.notify_changed()
    presets = [mv.ammo_combo.itemText(i) for i in range(mv.ammo_combo.count())]
    assert presets == ["Custom 62gr", "LC M855 (5.56)"]


def test_malformed_ammo_config_does_not_crash_launch(tmp_path, monkeypatch, qtbot):
    from PySide6 import QtWidgets

    from sound_metric_app.ui import main_window as mw

    # A hand-edited config with a non-list ammo_definitions makes
    # config.get_ammo_definitions raise ValueError. That read happens during
    # MainWindow.__init__ (notify_changed -> MarkingView.refresh -> _populate_ammo),
    # so it must surface as a dialog, not an unhandled traceback that stops launch.
    config = tmp_path / "sma_config.json"
    config.write_text('{"ammo_definitions": "LC M855"}', encoding="utf-8")
    monkeypatch.setenv("SMA_CONFIG", str(config))

    shown: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *a, **k: shown.append(a[2])
    )

    controller = WorkflowController(tmp_path / "wf.db")
    win = mw.MainWindow(controller)  # must not raise
    qtbot.addWidget(win)

    assert shown and "ammo_definitions" in shown[0]
    mv = win.marking_view
    assert mv.ammo_combo.count() == 0


def test_ammo_definitions_dialog_add_and_remove(window):
    from sound_metric_app.ui.main_window import AmmoDefinitionsDialog

    dialog = AmmoDefinitionsDialog(["LC M193 (5.56)"], parent=window)
    # Add a new type; a duplicate of an existing one is ignored.
    dialog.entry.setText("Custom 62gr")
    dialog._add()
    dialog.entry.setText("LC M193 (5.56)")
    dialog._add()
    assert dialog.definitions() == ["LC M193 (5.56)", "Custom 62gr"]

    # Remove the first, selected item.
    dialog.list.setCurrentRow(0)
    dialog._remove()
    assert dialog.definitions() == ["Custom 62gr"]
