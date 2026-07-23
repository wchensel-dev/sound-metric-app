"""Qt smoke/acceptance test for the workflow window.

Drives ingest -> mark -> bring-forward -> report through the real widgets
(buttons, combos, trees), so the off-thread task wiring and cross-view refresh
are exercised, not just the controller. Skipped when the ``gui`` extra is absent.

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
    # One string of fire: Dewesoft counts from zero, so 0000 is the FRP and
    # 0001 a regular.
    for name in ("SUP-1_AR15_01_0000.dxd", "SUP-1_AR15_01_0001.dxd"):
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


def test_report_empty_slot_row_spans_all_columns(window, monkeypatch):
    # A slot with nothing included renders a "none included" placeholder row
    # rather than being hidden — a missing quadrant is information. It must
    # carry a cell for every column so it stays aligned with the header (and
    # keeps tracking _METRIC_KEYS if the metric set grows again).
    from sound_metric_app.models import Batch, Combination
    from sound_metric_app.services.aggregation_service import BatchAverages
    from sound_metric_app.services.inclusion_service import InclusionService

    rv = window.report_view
    with window.controller._repo() as repo:
        combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
        batch_id = repo.create_batch(combination_id)
        status = InclusionService(repo).status(batch_id)

    report = BatchAverages(
        batch=Batch(combination_id=combination_id, id=batch_id),
        combination=Combination(sku="SUP-1", platform="AR15", ammo="M855", id=combination_id),
        n_shots=0,
        averages={},
        shots={},
        status=status,
    )
    monkeypatch.setattr(rv.controller, "batch_averages", lambda _batch_id: report)
    rv.batch_combo.blockSignals(True)
    rv.batch_combo.addItem("#1", batch_id)  # give _load_report a non-None batch id
    rv.batch_combo.blockSignals(False)

    rv._load_report()

    # All four slots are listed, every one flagged as empty.
    assert rv.tree.topLevelItemCount() == 4
    item = rv.tree.topLevelItem(0)
    # Every column has a cell -> nothing shifts left; the row matches the header.
    assert item.columnCount() == len(rv._COLUMNS)
    assert item.text(0) == "Muzzle Left · FRP"
    assert item.text(rv._FIRST_METRIC_COL) == "none included"
    assert "0 of 0 shot(s) brought forward" in rv.status_label.text()


def test_window_builds_with_four_tabs(window):
    assert window.tabs.count() == 4
    assert [window.tabs.tabText(i) for i in range(4)] == [
        "Ingest",
        "Mark",
        "Data bank",
        "Batch average",
    ]


def test_selecting_shot_with_null_keys_does_not_crash_mark_tab(window):
    # A shot whose filename yielded no placement keys is stored with
    # suppressor_sku/test_platform/cluster_index = None. Selecting it must not
    # pass None to QLineEdit.setPlaceholderText (which raises TypeError).
    with window.controller._repo() as repo:
        shot_id = repo.add_unmarked_shot("no-keys.dxd", None, None, None, None)

    mv = window.marking_view
    mv.refresh()  # _on_shot_changed fires on selection; must not raise
    mv.shot_combo.setCurrentIndex(mv._index_of_shot(shot_id))

    assert mv._current_shot_id() == shot_id
    assert mv.sku_edit.placeholderText() == ""
    assert mv.platform_edit.placeholderText() == ""
    assert mv.cluster_edit.placeholderText() == ""
    # No order means no derivable role, shown as an em-dash rather than a guess.
    assert mv.role_label.text() == "—"


def test_mark_form_previews_the_derived_role(window, qtbot):
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    mv = window.marking_view
    mv.refresh()

    # Blank box: the role falls back to the order the filename supplied.
    mv.shot_combo.setCurrentIndex(0)
    assert mv.role_label.text() == "FRP"

    # Typing an order re-derives it live; role is never entered by hand.
    mv.shot_order_edit.setText("4")
    assert mv.role_label.text() == "Regular"
    mv.shot_order_edit.setText("0")
    assert mv.role_label.text() == "FRP"
    mv.shot_order_edit.setText("not a number")
    assert mv.role_label.text() == "—"


def test_ingest_table_shows_cluster_and_role(window, qtbot):
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    table = window.ingest_view.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert "Cluster" in headers and "Role" in headers
    cluster_col, role_col = headers.index("Cluster"), headers.index("Role")
    assert table.item(0, cluster_col).text() == "1"
    assert table.item(0, role_col).text() == "FRP"
    assert table.item(1, role_col).text() == "Regular"


def _mark_all_shots(window, qtbot):
    """Ingest the fixture inbox and mark every shot (auto-tagged AI 1 / AI 2)."""
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    while window.ingest_view.table.rowCount() > 0:
        first_id = int(window.ingest_view.table.item(0, 0).text())
        window.open_marking_for(first_id)
        mv = window.marking_view
        qtbot.waitUntil(
            lambda: mv.ml_combo.isEnabled() and mv.ml_combo.count() >= 3, timeout=5000
        )
        mv.ammo_combo.setCurrentText("M855")
        before = window.ingest_view.table.rowCount()
        mv._mark()
        qtbot.waitUntil(
            lambda: window.ingest_view.table.rowCount() < before, timeout=5000
        )


def _include_everything(window):
    """Bring every marked shot forward so the batch-average view has data."""
    for shot in window.controller.shots_for_batch(window.controller.batches()[0].id):
        window.controller.include_shot(shot.id)


def test_clicking_metric_cell_graphs_that_shot(window, qtbot):
    from sound_metric_app.models import MicPosition

    _mark_all_shots(window, qtbot)
    _include_everything(window)

    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() == 4, timeout=5000)

    # Drill into a populated Shooter's Ear slot and grab one of its shot children.
    se_top = next(
        rv.tree.topLevelItem(i)
        for i in range(rv.tree.topLevelItemCount())
        if rv.tree.topLevelItem(i).text(0).startswith("Shooter's Ear")
        and rv.tree.topLevelItem(i).childCount() > 0
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


def test_window_start_label_says_when_no_onset_was_detected(qtbot):
    # A frame with no sample above the onset threshold falls back to the frame
    # start. The line still gets drawn (it is where the metric was computed from),
    # but labelling it plain "calc window starts" would read as a detected onset
    # at 0 ms on a silent or mis-triggered capture.
    import pyqtgraph as pg

    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.main_window import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)

    def start_label():
        for item in graph._plot.getPlotItem().items:
            if isinstance(item, pg.InfiniteLine) and item.label is not None:
                if "starts" in item.label.format:
                    return item.label.format
        return None

    trace = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        values=np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        y_label="SPL (dB)",
        title="Peak dB",
        window_start_index=0,
        window_end_index=3,
        onset_detected=False,
    )
    graph.show_trace(trace)
    assert start_label() == "calc window starts (no onset detected)"

    trace.onset_detected = True
    graph.show_trace(trace)
    assert start_label() == "calc window starts"


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
    from sound_metric_app.models import MicPosition, ShotRole

    # --- Ingest (off-thread) -> two unmarked rows ---
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)

    # --- Mark both shots via the marking form (channels auto-tag) ---
    _mark_all_shots(window, qtbot)
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 0, timeout=5000)

    # --- Data bank shows the whole tree, everything idle ---
    bv = window.bank_view
    bv.refresh()
    combination_item = bv.tree.topLevelItem(0)
    assert combination_item.text(0) == "SUP-1 / AR15 / M855"
    batch_item = combination_item.child(0)
    cluster_item = batch_item.child(0)
    assert cluster_item.childCount() == 2
    assert all(
        cluster_item.child(i).checkState(0) == QtCore.Qt.Unchecked for i in range(2)
    )
    assert "FRP: 0/3" in batch_item.text(1)

    # --- Nothing is averaged until brought forward ---
    rv = window.report_view
    rv.refresh()
    assert "0 of 2 shot(s) brought forward" in rv.status_label.text()

    # --- Bring the cluster forward from the tree ---
    bv.tree.setCurrentItem(cluster_item)
    assert bv.include_btn.isEnabled()
    bv._set_inclusion(True)
    qtbot.waitUntil(
        lambda: window.controller.inclusion_status(
            window.controller.batches()[0].id
        ).progress[ShotRole.FRP].included == 1,
        timeout=5000,
    )

    # --- The four slots now report, positions and roles never mixed ---
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() == 4, timeout=5000)
    labels = [rv.tree.topLevelItem(i).text(0) for i in range(4)]
    assert labels == [
        "Muzzle Left · FRP",
        "Muzzle Left · Regular",
        "Shooter's Ear · FRP",
        "Shooter's Ear · Regular",
    ]
    assert "2 of 2 shot(s) brought forward" in rv.status_label.text()
    # Each populated slot expands to the individual shots behind it.
    assert all(rv.tree.topLevelItem(i).childCount() == 1 for i in range(4))
    report = window.controller.batch_averages(window.controller.batches()[0].id)
    assert report.averages[(MicPosition.ML, ShotRole.FRP)]["n"] == 1

    # --- Close the batch from the tree ---
    bv.refresh()
    bv.tree.setCurrentItem(bv.tree.topLevelItem(0).child(0))
    assert bv.close_btn.isEnabled()


def _tree_nodes(bv):
    """The (combination, batch, cluster, shot) items of a single-branch tree."""
    combination = bv.tree.topLevelItem(0)
    batch = combination.child(0)
    cluster = batch.child(0)
    return combination, batch, cluster, cluster.child(0)


def test_action_buttons_track_the_selected_level(window, qtbot):
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    combination_item, batch_item, cluster_item, shot_item = _tree_nodes(bv)

    bv.tree.setCurrentItem(combination_item)
    # A combination is a container, not a roll-up unit or an editable session.
    assert not bv.include_btn.isEnabled()
    assert not bv.edit_btn.isEnabled()

    bv.tree.setCurrentItem(batch_item)
    assert bv.edit_btn.isEnabled()  # batch: session metadata
    assert bv.close_btn.isEnabled()
    assert not bv.include_btn.isEnabled()

    bv.tree.setCurrentItem(cluster_item)
    assert bv.include_btn.isEnabled()  # cluster: bring the whole string forward
    assert not bv.edit_btn.isEnabled()

    bv.tree.setCurrentItem(shot_item)
    assert bv.include_btn.isEnabled() and bv.exclude_btn.isEnabled()
    assert bv.edit_btn.isEnabled()  # shot: re-mark


def test_shot_checkbox_toggles_inclusion(window, qtbot):
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)
    assert shot.included is False

    # Ticking the box is the bring-forward gesture; it must persist.
    shot_item.setCheckState(0, QtCore.Qt.Checked)
    qtbot.waitUntil(lambda: window.controller.get_shot(shot.id).included, timeout=5000)

    # And a refresh must not fire the handler for the states it writes itself.
    bv.refresh()
    _, _, _cluster, refreshed = _tree_nodes(bv)
    assert refreshed.checkState(0) == QtCore.Qt.Checked
    assert window.controller.get_shot(shot.id).included is True


def test_checkbox_write_runs_after_the_signal_unwinds(window, qtbot):
    """The toggle must not rebuild the tree from inside itemChanged.

    Writing inline refreshes the view, which clears the tree and frees the very
    row Qt is still emitting itemChanged for — a use-after-free that crashes the
    application. So the write is deferred: nothing may reach the database until
    the event loop turns.
    """
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)

    shot_item.setCheckState(0, QtCore.Qt.Checked)
    assert window.controller.get_shot(shot.id).included is False  # still deferred
    qtbot.waitUntil(lambda: window.controller.get_shot(shot.id).included, timeout=5000)


def test_a_rejected_toggle_snaps_the_checkbox_back(window, qtbot, monkeypatch):
    """A shot with no order has no role, so ticking it fails — and must not lie.

    The row is left showing the flag that was actually stored, not the tick the
    user made, with the reason surfaced in a dialog.
    """
    from PySide6 import QtWidgets

    _mark_all_shots(window, qtbot)
    bv = window.bank_view
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)
    with window.controller._repo() as repo:
        repo._conn.execute("UPDATE shots SET shot_order = NULL WHERE id = ?", (shot.id,))
        repo._conn.commit()
    bv.refresh()
    _, _, _cluster, shot_item = _tree_nodes(bv)

    errors = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a))
    )
    shot_item.setCheckState(0, QtCore.Qt.Checked)
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)

    assert window.controller.get_shot(shot.id).included is False
    _, _, _cluster, refreshed = _tree_nodes(bv)
    assert refreshed.checkState(0) == QtCore.Qt.Unchecked


def test_exclude_prompts_for_a_reason_and_records_it(window, qtbot, monkeypatch):
    from PySide6 import QtWidgets

    _mark_all_shots(window, qtbot)
    _include_everything(window)

    bv = window.bank_view
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)
    bv.tree.setCurrentItem(shot_item)

    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", staticmethod(lambda *a, **k: ("high winds", True))
    )
    bv._set_inclusion(False)

    stored = window.controller.get_shot(shot.id)
    assert stored.included is False and stored.exclusion_reason == "high winds"
    # The reason surfaces in the tree so an excluded shot explains itself.
    bv.refresh()
    _, _, _cluster, refreshed = _tree_nodes(bv)
    assert "high winds" in refreshed.text(1)


def test_edit_batch_session_metadata_via_tree(window, qtbot, monkeypatch):
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _combination_item, batch_item, *_rest = _tree_nodes(bv)
    bv.tree.setCurrentItem(batch_item)

    from PySide6 import QtWidgets

    from sound_metric_app.ui.main_window import BatchEditDialog

    # Stand in for the modal: fill the session form and accept it.
    def fake_exec(self):
        self.label_edit.setText("Morning string")
        self.date_edit.setText("2026-07-22")
        self.wind_edit.setText("4")
        self.notes_edit.setPlainText("clear, light crosswind")
        self._on_accept()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(BatchEditDialog, "exec", fake_exec)
    bv._edit_selected()

    batch = window.controller.batches()[0]
    assert batch.label == "Morning string" and batch.session_date == "2026-07-22"
    assert batch.wind_speed == 4.0 and batch.notes == "clear, light crosswind"
    _combination_item, batch_item, *_rest = _tree_nodes(bv)
    assert "Morning string 2026-07-22" in batch_item.text(0)


def test_batch_edit_dialog_rejects_a_malformed_date(window, monkeypatch):
    from PySide6 import QtWidgets

    from sound_metric_app.models import Batch
    from sound_metric_app.ui.main_window import BatchEditDialog

    warned: list = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))

    dialog = BatchEditDialog(Batch(id=1, combination_id=1), combination_label="X", parent=window)
    dialog.date_edit.setText("22-07-2026")
    dialog._on_accept()

    assert warned and "YYYY-MM-DD" in warned[0]
    assert dialog.result() != QtWidgets.QDialog.Accepted


def test_double_click_edits_leaf_and_batch_rows_only(window, qtbot, monkeypatch):
    # Double-click is Qt's expand/collapse gesture on container rows; it must not
    # also route through _edit_selected on a combination or cluster.
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    edited: list = []
    monkeypatch.setattr(bv, "_edit_selected", lambda: edited.append(True))

    combination_item, batch_item, cluster_item, shot_item = _tree_nodes(bv)

    bv._on_item_double_clicked(combination_item, 0)
    bv._on_item_double_clicked(cluster_item, 0)
    qtbot.wait(50)
    assert edited == []  # pure containers: no edit modal

    # Deferred out of the double-click emission (saving an edit refreshes the
    # tree, which would free the row Qt is still emitting for), so wait for it.
    bv._on_item_double_clicked(batch_item, 0)
    bv._on_item_double_clicked(shot_item, 0)
    qtbot.waitUntil(lambda: edited == [True, True], timeout=5000)


def test_edit_shot_re_marks_with_corrected_ammo(window, qtbot):
    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    first_id = int(window.ingest_view.table.item(0, 0).text())
    window.open_marking_for(first_id)
    mv = window.marking_view
    qtbot.waitUntil(lambda: mv.ml_combo.isEnabled() and mv.ml_combo.count() >= 3, timeout=5000)
    mv.ammo_combo.setCurrentText("WRONG")
    mv._mark()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 1, timeout=5000)

    bv = window.bank_view
    bv.refresh()
    _combination_item, _batch_item, cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, cluster, batch, combo = shot_item.data(0, QtCore.Qt.UserRole)

    # Open the pre-filled dialog directly (bypassing the async channel load),
    # correct the ammo, and accept it as the user would.
    from sound_metric_app.ui.main_window import ShotEditDialog

    dialog = ShotEditDialog(
        shot,
        sku=combo.sku,
        platform=combo.platform,
        ammo=combo.ammo,
        cluster_index=cluster.cluster_index,
        channel_names=["AI 1", "AI 2"],
        parent=bv,
    )
    # Pre-filled from where the shot actually landed, not its filename keys.
    assert dialog.ammo_combo.currentText() == "WRONG"
    assert dialog.ml_combo.currentText() == "AI 1"  # from the auto-tagged shot
    assert dialog.se_combo.currentText() == "AI 2"
    assert dialog.cluster_edit.text() == "1"
    assert dialog.role_label.text() == "FRP"
    dialog.ammo_combo.setCurrentText("M855")
    dialog._on_accept()

    bv._run_async(
        lambda: window.controller.mark(shot.id, **dialog.values()),
        lambda _r: window.notify_changed(),
    )
    # The shot moves to the corrected combination; the emptied "WRONG" branch is
    # swept so the tree does not keep an empty combination behind.
    qtbot.waitUntil(
        lambda: {c.ammo for c in window.controller.combinations()} == {"M855"},
        timeout=5000,
    )
    combinations = window.controller.combinations()
    assert len(combinations) == 1
    tree = window.controller.data_bank()
    assert tree[0].batches[0].clusters[0].shots[0].id == shot.id


def test_shot_edit_dialog_requires_a_cluster(window, monkeypatch):
    from PySide6 import QtWidgets

    from sound_metric_app.models import Shot
    from sound_metric_app.ui.main_window import ShotEditDialog

    warned: list = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))

    dialog = ShotEditDialog(
        Shot(source_file="f.dxd", shot_order=1, ml_channel="AI 1"),
        sku="SUP-1",
        platform="AR15",
        ammo="M855",
        cluster_index=None,
        channel_names=["AI 1", "AI 2"],
        parent=window,
    )
    dialog._on_accept()

    # Without a cluster there is no string of fire to place the shot in.
    assert warned and "cluster" in warned[0].lower()
    assert dialog.result() != QtWidgets.QDialog.Accepted


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
