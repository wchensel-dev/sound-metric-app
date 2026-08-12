"""Qt smoke/acceptance test for the workflow window.

Drives ingest -> mark -> bring-forward -> report through the real widgets
(buttons, combos, trees), so the off-thread task wiring and cross-view refresh
are exercised, not just the controller. Skipped when the ``gui`` extra is absent.

Run headless:  QT_QPA_PLATFORM=offscreen pytest tests/test_ui.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets  # noqa: E402

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
    from sound_metric_app.ui.format import _format_metric

    assert _format_metric(None) == "—"
    assert _format_metric(163.4) == "163.40"
    assert _format_metric(0) == "0.00"


def test_report_column_indices_stay_in_step_with_the_metric_set():
    # The column constants are derived from _METRIC_KEYS and _COLUMNS, so they
    # only agree while those two lists agree with each other. Adding a metric key
    # without its header (or slipping a column in between the metrics and the
    # trailing button ones) would silently either push a real metric past the
    # _on_cell_clicked guard or aim _METRIC_KEYS at the wrong column — neither of
    # which raises. Pin the layout instead.
    from sound_metric_app.ui.views.batch_average import BatchAverageView as bv

    # Label, n, every metric, every diagnostic, then exactly the two trailing
    # button columns.
    assert len(bv._COLUMNS) == (
        bv._FIRST_METRIC_COL + len(bv._METRIC_KEYS) + len(bv._DIAGNOSTIC_KEYS) + 2
    )
    # The diagnostics sit *past* _END_METRIC_COL, which is the guard's upper
    # bound in _on_cell_clicked: that is what stops a click on one being read as
    # a request to graph a metric it has no trace for. Compare then follows the
    # diagnostics, and paste follows Compare, so "Compare sits before Copy" and
    # "diagnostics are not graphable" are both properties of the constants.
    assert bv._END_METRIC_COL == bv._FIRST_METRIC_COL + len(bv._METRIC_KEYS)
    assert bv._COMPARE_COL == bv._END_METRIC_COL + len(bv._DIAGNOSTIC_KEYS)
    assert bv._PASTE_COL == bv._COMPARE_COL + 1
    assert bv._COLUMNS[bv._COMPARE_COL] == "Compare"
    assert bv._COLUMNS[bv._PASTE_COL] == "Scout paste"


def test_report_columns_and_compare_metrics_come_from_one_list():
    # The Batch average tree's metric columns and the Compare tab's metric picker
    # are the same set spent two ways. Deriving both from _REPORT_METRICS is what
    # stops a metric added to one from going missing on the other.
    from sound_metric_app.ui.metric_columns import _REPORT_METRICS
    from sound_metric_app.ui.views.batch_average import BatchAverageView as bv

    assert bv._METRIC_KEYS == tuple(key for _label, key in _REPORT_METRICS)
    assert bv._COLUMNS[bv._FIRST_METRIC_COL : bv._END_METRIC_COL] == [
        label for label, _key in _REPORT_METRICS
    ]


def test_diagnostic_columns_are_not_graphable_metrics():
    # The diagnostics are a separate list precisely because every _REPORT_METRICS
    # key must be one build_metric_trace accepts — the Compare picker and the
    # click-to-graph handler both assume it. Moving a diagnostic into that list
    # would raise ValueError on the first click, so keep the two disjoint and
    # keep the diagnostics out of the graphable range.
    from sound_metric_app.dsp import build_metric_trace
    from sound_metric_app.ui.metric_columns import _REPORT_DIAGNOSTICS, _REPORT_METRICS
    from sound_metric_app.ui.views.batch_average import BatchAverageView as bv

    metric_keys = {key for _label, key in _REPORT_METRICS}
    diagnostic_keys = {key for _label, key in _REPORT_DIAGNOSTICS}
    assert metric_keys.isdisjoint(diagnostic_keys)

    frame = Frame(samples=np.zeros(64), sample_rate=1000.0, source_file="f", channel="c")
    for key in diagnostic_keys:
        with pytest.raises(ValueError):
            build_metric_trace(frame, key)

    # ...and they render past the graph guard's upper bound.
    assert bv._COLUMNS[bv._END_METRIC_COL : bv._COMPARE_COL] == [
        label for label, _key in _REPORT_DIAGNOSTICS
    ]


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
    # Nothing was averaged, so there is nothing to paste: no Copy button.
    assert rv.tree.itemWidget(item, rv._PASTE_COL) is None


def test_report_slot_rows_render_an_amber_wash(window, qtbot):
    # Slot (averaging) rows carry a translucent amber wash so they read apart
    # from the individual shots nested under them. Asserted on rendered pixels,
    # not on the item's brush: the tree's stylesheet paints the row background
    # itself and discards a QTreeWidgetItem brush, so a brush-level assertion
    # passes while the app still shows plain grey/white.
    from sound_metric_app.ui.qtwidgets import _AVERAGE_ROW_TINT

    assert _AVERAGE_ROW_TINT.alpha() < 255  # translucent -> composites over base

    rv = window.report_view
    # Show the tab through the window first, so the splitter lays the tree out
    # at a real size (grabbing an unlaid-out child yields a 0x0 image) and so
    # the tab's own refresh() -- which clears the tree -- is already done.
    window.tabs.setCurrentWidget(rv)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)

    rv.tree.clear()
    parent = rv._add_top(["slot"] * len(rv._COLUMNS))
    child = QtWidgets.QTreeWidgetItem(["shot"] * len(rv._COLUMNS))
    parent.addChild(child)
    rv.tree.expandAll()
    # Grab the viewport, not the whole tree: visualItemRect is in viewport
    # coordinates, so this needs no header/frame offset arithmetic.
    pixmap = rv.tree.viewport().grab()
    image = pixmap.toImage()
    ratio = pixmap.devicePixelRatio()  # a HiDPI grab is larger than the widget

    def pixel(item):
        # Mid-row vertically; horizontally mid-viewport, which lands in the
        # blank part of the wide first column -- past the short label text, so
        # we read background rather than a glyph.
        row = rv.tree.visualItemRect(item)
        x = int(rv.tree.viewport().width() // 2 * ratio)
        return image.pixelColor(x, int(row.center().y() * ratio))

    slot_bg, shot_bg = pixel(parent), pixel(child)
    assert slot_bg != shot_bg  # the wash is actually visible
    # Amber, not an arbitrary shift: warmer toward red than blue vs. the plain row.
    assert slot_bg.red() >= shot_bg.red() and slot_bg.blue() < shot_bg.blue()


def test_window_builds_with_five_tabs(window):
    assert window.tabs.count() == 5
    assert [window.tabs.tabText(i) for i in range(5)] == [
        "Ingest",
        "Mark",
        "Data bank",
        "Batch average",
        "Compare",
    ]


def _seed_marked_shot(window, tmp_path, name):
    """Ingest + mark one capture through the controller (real DSP over sine frames).

    A fully marked shot gives the SKU-filter tests a real combination/batch/
    cluster/shot path to filter on.
    """
    folder = tmp_path / "seed"
    folder.mkdir(exist_ok=True)
    (folder / name).write_bytes(b"")
    window.controller.ingest(folder, validate=False)
    shot = next(s for s in window.controller.unmarked_shots() if s.source_file.endswith(name))
    window.controller.mark(shot.id, ammo="M855")


def test_data_bank_sku_filter_narrows_the_tree(window, tmp_path):
    _seed_marked_shot(window, tmp_path, "SUP-1_AR15_01_0000.dxd")
    _seed_marked_shot(window, tmp_path, "SUP-2_AR15_01_0000.dxd")

    bv = window.bank_view
    bv.refresh()

    # The dropdown is the "All SKUs" sentinel (data None) plus one row per SKU.
    labels = [bv.sku_combo.itemText(i) for i in range(bv.sku_combo.count())]
    assert labels == ["All SKUs", "SUP-1", "SUP-2"]
    assert bv.sku_combo.itemData(0) is None

    # All SKUs: the whole archive, both combinations at the top level.
    assert bv.tree.topLevelItemCount() == 2

    # Pick SUP-1: only its combination is built into the tree.
    bv.sku_combo.setCurrentIndex(labels.index("SUP-1"))
    assert bv.tree.topLevelItemCount() == 1
    assert bv.tree.topLevelItem(0).text(0).startswith("SUP-1")


def test_repopulate_sku_filter_preserves_selection_and_falls_back(qtbot):
    # The two branches _repopulate_sku_filter turns on, tested on the bare combo:
    # a still-present selection survives a rebuild; a swept-away one falls back to
    # the "All SKUs" sentinel rather than silently landing on the wrong SKU.
    from PySide6 import QtWidgets

    from sound_metric_app.ui.qtwidgets import _repopulate_sku_filter

    combo = QtWidgets.QComboBox()
    qtbot.addWidget(combo)

    _repopulate_sku_filter(combo, ["SUP-1", "SUP-2"])
    assert [combo.itemText(i) for i in range(combo.count())] == ["All SKUs", "SUP-1", "SUP-2"]
    assert combo.currentData() is None  # defaults to the no-filter sentinel

    # Select SUP-2, then rebuild with it still present: the selection is kept.
    combo.setCurrentIndex(2)
    assert combo.currentData() == "SUP-2"
    _repopulate_sku_filter(combo, ["SUP-1", "SUP-2", "SUP-3"])
    assert combo.currentData() == "SUP-2"

    # Rebuild with SUP-2 gone (its last combination was swept): fall back to All SKUs.
    _repopulate_sku_filter(combo, ["SUP-1", "SUP-3"])
    assert combo.currentIndex() == 0
    assert combo.currentData() is None


def test_repopulate_sku_filter_blocks_signals_over_the_rebuild(qtbot):
    # The rebuild must not fire currentIndexChanged — the caller re-reads the
    # selection and refreshes explicitly, so a signal here would double-refresh
    # (or refresh mid-rebuild against a half-filled combo).
    from PySide6 import QtWidgets

    from sound_metric_app.ui.qtwidgets import _repopulate_sku_filter

    combo = QtWidgets.QComboBox()
    qtbot.addWidget(combo)
    _repopulate_sku_filter(combo, ["SUP-1", "SUP-2"])
    combo.setCurrentIndex(2)

    fired: list = []
    combo.currentIndexChanged.connect(lambda _i: fired.append(True))
    # SUP-2 vanishes, so the current index genuinely changes (2 -> 0) during the
    # rebuild; without the signal block that transition would emit.
    _repopulate_sku_filter(combo, ["SUP-1"])
    assert fired == []


def test_data_bank_filter_change_does_not_prune_empty_containers(window, tmp_path):
    # Changing the SKU filter is pure navigation: refresh() must not run the
    # destructive sweep, so a transiently-empty cluster survives a filter change.
    _seed_marked_shot(window, tmp_path, "SUP-1_AR15_01_0000.dxd")
    bv = window.bank_view
    with window.controller._repo() as repo:
        batch = window.controller.batches()[0]
        stray = repo.upsert_cluster(batch.id, 9)

    bv.refresh()
    labels = [bv.sku_combo.itemText(i) for i in range(bv.sku_combo.count())]
    bv.sku_combo.setCurrentIndex(labels.index("SUP-1"))  # fires refresh via the combo

    tree = window.controller.data_bank()
    cluster_ids = {c.cluster.id for n in tree for b in n.batches for c in b.clusters}
    assert stray in cluster_ids


def test_notify_changed_prunes_empty_containers(window, tmp_path):
    # A mutation is where orphans are created, so notify_changed() (the post-mutation
    # funnel) is where the sweep runs and the stray empty cluster is dropped.
    _seed_marked_shot(window, tmp_path, "SUP-1_AR15_01_0000.dxd")
    with window.controller._repo() as repo:
        batch = window.controller.batches()[0]
        stray = repo.upsert_cluster(batch.id, 9)

    window.notify_changed()

    tree = window.controller.data_bank()
    cluster_ids = {c.cluster.id for n in tree for b in n.batches for c in b.clusters}
    assert stray not in cluster_ids


def test_batch_average_sku_filter_narrows_the_batch_list(window, tmp_path):
    _seed_marked_shot(window, tmp_path, "SUP-1_AR15_01_0000.dxd")
    _seed_marked_shot(window, tmp_path, "SUP-2_AR15_01_0000.dxd")

    rv = window.report_view
    rv.refresh()

    labels = [rv.sku_combo.itemText(i) for i in range(rv.sku_combo.count())]
    assert labels == ["All SKUs", "SUP-1", "SUP-2"]

    # All SKUs: both sessions listed in the batch picker.
    assert rv.batch_combo.count() == 2

    # Filter to SUP-2: only its session survives, and it is the one shown.
    rv.sku_combo.setCurrentIndex(labels.index("SUP-2"))
    assert rv.batch_combo.count() == 1
    assert "SUP-2" in rv.batch_combo.itemText(0)


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


def test_ingest_discard_and_restore_bad_file(window, tmp_path, qtbot):
    inbox = tmp_path / "inbox"
    bad = inbox / "not-a-valid-name.dxd"
    bad.write_bytes(b"")
    source_file = str(bad.resolve())

    view = window.ingest_view
    view._ingest()
    qtbot.waitUntil(lambda: view.bad_files_tree.topLevelItemCount() == 1, timeout=5000)
    assert view.bad_files_tree.topLevelItem(0).text(0) == bad.name

    # _discard bypasses the confirmation dialog (_prompt_discard), matching how
    # other tests here drive controller-facing methods directly rather than
    # simulating a click through a modal.
    view._discard(source_file, "test reason")
    qtbot.waitUntil(lambda: view.bad_files_tree.topLevelItemCount() == 0, timeout=5000)
    qtbot.waitUntil(lambda: view.discarded_tree.topLevelItemCount() == 1, timeout=5000)
    assert view.discarded_tree.topLevelItem(0).text(1) == "test reason"

    view._restore(source_file)
    qtbot.waitUntil(lambda: view.discarded_tree.topLevelItemCount() == 0, timeout=5000)

    # Restoring doesn't retroactively re-ingest; the file only resurfaces as
    # malformed on the next explicit scan.
    view._ingest()
    qtbot.waitUntil(lambda: view.bad_files_tree.topLevelItemCount() == 1, timeout=5000)


def test_discard_unmarked_shot_from_ingest_and_mark_pages(window, qtbot):
    view = window.ingest_view
    mv = window.marking_view
    view._ingest()
    qtbot.waitUntil(lambda: view.table.rowCount() == 2, timeout=5000)
    mv.refresh()
    qtbot.waitUntil(lambda: mv.shot_combo.count() == 2, timeout=5000)

    first_id = int(view.table.item(0, 0).text())
    source_file = next(
        s.source_file for s in window.controller.unmarked_shots() if s.id == first_id
    )

    # Bypass the confirmation dialog (_prompt_discard_shot) and drive the
    # controller directly, same convention as the bad-file discard test.
    window.controller.discard_shot(first_id, reason="wrong recording")
    window.notify_changed()

    qtbot.waitUntil(lambda: view.table.rowCount() == 1, timeout=5000)
    qtbot.waitUntil(lambda: mv.shot_combo.count() == 1, timeout=5000)
    assert all(
        int(view.table.item(row, 0).text()) != first_id for row in range(view.table.rowCount())
    )
    qtbot.waitUntil(lambda: view.discarded_tree.topLevelItemCount() == 1, timeout=5000)
    assert view.discarded_tree.topLevelItem(0).text(0) == Path(source_file).name

    # A discarded shot's file must not resurface on the next scan.
    view._ingest()
    qtbot.waitUntil(lambda: view.table.rowCount() == 1, timeout=5000)


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
    assert rv.graph._framing._auto_frame_btn.isEnabled()
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
    assert not rv.graph._framing._auto_frame_btn.isEnabled()
    assert rv.graph._x_bounds is None

    # Same for the trailing Scout-paste column, which sits *past* the metrics:
    # indexing _METRIC_KEYS with it would raise rather than simply miss.
    rv._on_cell_clicked(shot_item, rv._PASTE_COL)
    assert rv._current_request is None
    assert len(rv.graph._plot.listDataItems()) == 0


def test_scout_paste_button_copies_the_row_it_sits_on(window, qtbot):
    # Every averaged row ends in a Copy button that puts that slot's SSR1 string
    # on the clipboard for the SilencerScout report editor. The string is only
    # trustworthy if it carries the numbers of the row it sits on -- their end
    # cannot tell a mispasted string from a good one.
    from sound_metric_app.ui.views.batch_average import _COPIED_LABEL, _COPY_LABEL

    _mark_all_shots(window, qtbot)
    _include_everything(window)

    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() == 4, timeout=5000)

    frp_se = next(
        rv.tree.topLevelItem(i)
        for i in range(rv.tree.topLevelItemCount())
        if rv.tree.topLevelItem(i).text(0) == "Shooter's Ear · FRP"
    )
    button = rv.tree.itemWidget(frp_se, rv._PASTE_COL)
    assert button is not None and button.text() == _COPY_LABEL
    # The string names neither the test nor the shot type's source, so the button
    # has to: the operator picks the receiving row by hand.
    assert "Shooter's Ear · FRP" in button.toolTip()
    assert "SUP-1 / AR15 / M855" in button.toolTip()

    button.click()
    line = QtWidgets.QApplication.clipboard().text()
    assert button.toolTip().endswith(line)  # the tooltip previewed what was copied

    tag, shot_type, field = line.split("|")
    mic, values = field.split("=")
    assert (tag, shot_type, mic) == ("SSR1", "frp", "SE")
    # Four values, in the order LIAeq100ms dB, Peak dB, Peak dBA, Impulse Pa·ms
    # -- and each one exactly what its column on this row shows.
    columns = ["LIAeq,100ms dBA", "Peak dB", "Peak dBA", "Impulse Pa·ms"]
    assert values.split(",") == [frp_se.text(rv._COLUMNS.index(c)) for c in columns]

    # A clipboard write is invisible, so the button confirms it and then reverts.
    assert button.text() == _COPIED_LABEL
    qtbot.waitUntil(lambda: button.text() == _COPY_LABEL, timeout=5000)


def test_scout_paste_button_survives_the_row_being_rebuilt(window, qtbot):
    # The revert timer outlives the click by a second, and reloading the report
    # deletes the button it points at. Anchoring the timer to the button is what
    # keeps that from firing into a deleted widget and taking the window with it.
    _mark_all_shots(window, qtbot)
    _include_everything(window)

    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() == 4, timeout=5000)
    button = rv.tree.itemWidget(rv.tree.topLevelItem(0), rv._PASTE_COL)
    button.click()
    rv._load_report()  # drops the tree, and with it the button just clicked

    qtbot.wait(1500)  # past _COPIED_FLASH_MS: the revert would have fired by now
    assert rv.tree.topLevelItemCount() == 4


def _frp_compare_buttons(rv):
    """The Compare button on the first shot row of each populated FRP slot.

    Both FRP slots hold the *same* shot (one capture carries both mics), so the
    pair is the ML and SE series of one gunshot — the overlay this feature was
    asked for.
    """
    return [
        rv.tree.itemWidget(top.child(0), rv._COMPARE_COL)
        for i in range(rv.tree.topLevelItemCount())
        if (top := rv.tree.topLevelItem(i)).text(0).endswith("FRP") and top.childCount()
    ]


def _row_button(cv, row: int, column: int):
    """The button in one Compare row's action column."""
    return cv.tree.itemWidget(cv.tree.topLevelItem(row), column)


def _loaded_report(window, qtbot):
    """Mark + include the fixture shots and return the loaded Batch average view."""
    _mark_all_shots(window, qtbot)
    _include_everything(window)
    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() == 4, timeout=5000)
    return rv


def test_compare_button_pins_a_shot_row_to_the_compare_tab(window, qtbot):
    # The Compare button sits on shot rows only, one column before Copy: an
    # average has no capture to draw a curve from, and a shot has no paste
    # string. Which mic it pins is the slot the row sits under, so pinning the
    # same shot from both FRP slots overlays its ML and SE curves.
    from sound_metric_app.models import MicPosition
    from sound_metric_app.ui.compare_series import (
        _COMPARE_ADDED_LABEL,
        _COMPARE_ALREADY_LABEL,
        _COMPARE_LABEL,
    )

    rv = _loaded_report(window, qtbot)
    cv = window.compare_view
    compare_tab = window.tabs.indexOf(cv)

    # An averaged (top-level) row has no Compare button; its shot rows do.
    assert rv.tree.itemWidget(rv.tree.topLevelItem(0), rv._COMPARE_COL) is None
    buttons = _frp_compare_buttons(rv)
    assert len(buttons) == 2

    for button in buttons:
        assert button.text() == _COMPARE_LABEL
        button.click()
        assert button.text() == _COMPARE_ADDED_LABEL  # pinning shows nowhere else

    assert [s.position for s in cv._series] == [MicPosition.ML, MicPosition.SE]
    assert len({s.shot_id for s in cv._series}) == 1  # one shot, both its mics
    # Pinning happens on this tab and draws on another, so the count travels
    # with the tab title.
    assert window.tabs.tabText(compare_tab) == "Compare (2)"
    # A repeat pin would draw a curve exactly on top of itself: refused, and said so.
    buttons[0].click()
    assert buttons[0].text() == _COMPARE_ALREADY_LABEL
    assert len(cv._series) == 2

    # Both curves are drawn, keyed by a legend, over the default Impulse metric.
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)
    assert cv.metric_combo.currentData() == "impulse_pa_ms"
    assert cv.graph._legend.isVisible()
    assert cv.tree.topLevelItemCount() == 2
    assert "2 of 2 drawn" in cv.status_label.text()
    # The rows are the legend's index: one per series, in the drawn order.
    assert [
        cv.tree.topLevelItem(i).text(cv._LABEL_COL) for i in range(2)
    ] == [s.label for s in cv._series]
    assert str(cv._series[0].shot_id) in cv._series[0].label
    assert cv._series[0].label.endswith("ML")


def test_compare_removes_and_clears_pinned_shots(window, qtbot):
    rv = _loaded_report(window, qtbot)
    cv = window.compare_view
    compare_tab = window.tabs.indexOf(cv)
    for button in _frp_compare_buttons(rv):
        button.click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)

    # Subtracting a shot: the row's own Remove takes it and its curve off. The
    # survivor's trace is already loaded, so this redraws with no capture read.
    _row_button(cv, 0, cv._REMOVE_COL).click()
    # Deferred a turn of the event loop: the handler rebuilds the very tree the
    # button lives in, so it must not run inside the click it was clicked by.
    qtbot.waitUntil(lambda: len(cv._series) == 1, timeout=5000)
    assert [s.position.value for s in cv._series] == ["SE"]
    assert len(cv.graph._plot.listDataItems()) == 1
    assert not cv.graph._legend.isVisible()  # one curve needs no key
    assert window.tabs.tabText(compare_tab) == "Compare (1)"

    cv.clear()
    assert cv._series == []
    assert cv.tree.topLevelItemCount() == 0
    assert len(cv.graph._plot.listDataItems()) == 0
    assert window.tabs.tabText(compare_tab) == "Compare"


def test_compare_row_gives_its_width_to_the_label_not_the_buttons(window, qtbot):
    # A tree header stretches its *last* section by default, which handed the
    # spare width to the Remove column and elided every row down to "#7 ·…".
    # The label is what identifies a row, so the stretch belongs to it and the
    # button columns stay at what a button needs.
    cv = window.compare_view
    header = cv.tree.header()

    assert not header.stretchLastSection()
    assert header.sectionResizeMode(cv._LABEL_COL) == QtWidgets.QHeaderView.Stretch
    for col in (cv._HIDE_COL, cv._REMOVE_COL):
        assert header.sectionResizeMode(col) == QtWidgets.QHeaderView.Fixed
        # Sized from a real button, so a larger system font widens the column
        # instead of spilling out of it.
        assert cv.tree.columnWidth(col) >= QtWidgets.QPushButton("Remove").sizeHint().width()


def test_compare_hide_takes_a_curve_off_without_unpinning_it(window, qtbot):
    # Hide is the "read three of these five" control: the curve leaves the
    # graph, the row stays, and the colours of everything still drawn hold
    # still -- a reshuffle on every toggle would undo the legend the operator
    # has just learned.
    rv = _loaded_report(window, qtbot)
    cv = window.compare_view
    compare_tab = window.tabs.indexOf(cv)
    for button in _frp_compare_buttons(rv):
        button.click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)
    second_color = cv.graph._series[1][2]

    assert _row_button(cv, 0, cv._HIDE_COL).text() == "Hide"
    _row_button(cv, 0, cv._HIDE_COL).click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 1, timeout=5000)

    # Still pinned: same rows, same count on the tab, nothing unloaded.
    assert len(cv._series) == 2
    assert cv.tree.topLevelItemCount() == 2
    assert window.tabs.tabText(compare_tab) == "Compare (2)"
    assert len(cv._traces) == 2  # the hidden curve is kept, so unhiding is free
    assert "1 of 2 drawn" in cv.status_label.text()
    assert "1 hidden" in cv.status_label.text()
    # The survivor kept its colour rather than sliding up to the first one.
    assert cv.graph._series[0][2] == second_color

    # The button now offers the way back, and takes it.
    assert _row_button(cv, 0, cv._HIDE_COL).text() == "Show"
    _row_button(cv, 0, cv._HIDE_COL).click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)
    assert cv._hidden == set()
    assert "2 of 2 drawn" in cv.status_label.text()
    assert cv.graph._series[1][2] == second_color


def test_compare_hidden_shot_is_not_read_back_off_disk(window, qtbot):
    # A hidden curve is not drawn, so its capture is not worth re-reading when
    # the metric changes. Unhiding is what asks for it.
    rv = _loaded_report(window, qtbot)
    cv = window.compare_view
    for button in _frp_compare_buttons(rv):
        button.click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)
    _row_button(cv, 0, cv._HIDE_COL).click()
    qtbot.waitUntil(lambda: len(cv._hidden) == 1, timeout=5000)

    cv.metric_combo.setCurrentIndex(cv.metric_combo.findData("peak_db"))
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 1, timeout=5000)
    assert len(cv._traces) == 1  # only the shown one was re-read

    # Unhiding loads it, at the metric now selected.
    _row_button(cv, 0, cv._HIDE_COL).click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)
    assert cv.graph._plot.getAxis("left").labelText == "SPL (dB)"


def test_compare_keeps_its_framing_across_a_tab_switch(window, qtbot):
    # Pinning is done on the *other* tab, so the operator crosses back and forth
    # while assembling an overlay. A refresh that redrew would autorange away
    # whatever they had just framed -- so navigation leaves the graph alone, and
    # only an actual mutation (which can change what a curve is drawn from)
    # forces the reload.
    rv = _loaded_report(window, qtbot)
    cv = window.compare_view
    _frp_compare_buttons(rv)[0].click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 1, timeout=5000)

    window.tabs.setCurrentWidget(cv)
    cv.graph._framing._frame_onset_btns[0].click()
    framed = cv.graph._plot.getViewBox().viewRange()[0]

    window.tabs.setCurrentWidget(rv)
    window.tabs.setCurrentWidget(cv)
    assert cv.graph._plot.getViewBox().viewRange()[0] == pytest.approx(framed)
    assert len(cv.graph._plot.listDataItems()) == 1

    # A mutation drops the memoized curves, so the next refresh does reload them.
    window.notify_changed()
    assert cv._traces == {}
    qtbot.waitUntil(lambda: len(cv._traces) == 1, timeout=5000)
    assert len(cv.graph._plot.listDataItems()) == 1


def test_compare_metric_switch_redraws_every_pinned_shot(window, qtbot):
    # The metric is chosen once for the whole overlay -- curves have to share a Y
    # axis to be read against each other -- so switching it reloads all of them.
    rv = _loaded_report(window, qtbot)
    cv = window.compare_view
    for button in _frp_compare_buttons(rv):
        button.click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)
    assert cv.graph._plot.getAxis("left").labelText == "Impulse ∫p·dt (Pa·ms)"

    cv.metric_combo.setCurrentIndex(cv.metric_combo.findData("peak_dba"))
    qtbot.waitUntil(
        lambda: cv.graph._plot.getAxis("left").labelText == "SPL (dBA)", timeout=5000
    )
    assert len(cv.graph._plot.listDataItems()) == 2
    assert len(cv._traces) == 2  # the new metric's curves, not the old ones


def test_compare_draws_the_shots_it_can_and_flags_the_ones_it_cannot(window, qtbot):
    # A pinned shot whose capture has moved must not take the rest of the
    # overlay down with it -- and must not pop a dialog either, since every
    # redraw would pop it again. It is reported against its own row instead.
    rv = _loaded_report(window, qtbot)
    cv = window.compare_view
    for button in _frp_compare_buttons(rv):
        button.click()
    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 2, timeout=5000)

    good = cv._series[1]
    real_trace = cv.controller.metric_trace

    def flaky(shot_id, position, metric_key, **kwargs):
        if position is not good.position:
            raise FileNotFoundError("capture has moved")
        return real_trace(shot_id, position, metric_key, **kwargs)

    cv.controller.metric_trace = flaky
    cv.invalidate_traces()
    cv.refresh()

    qtbot.waitUntil(lambda: len(cv.graph._plot.listDataItems()) == 1, timeout=5000)
    row = cv.tree.topLevelItem(0)
    assert cv.tree.topLevelItemCount() == 2  # the failed shot keeps its row
    assert "unavailable" in row.text(cv._LABEL_COL)
    assert "capture has moved" in row.toolTip(cv._LABEL_COL)
    assert "1 of 2 drawn" in cv.status_label.text()
    assert "1 unavailable" in cv.status_label.text()
    # Its row still offers both actions -- an unloadable curve is still unpinnable.
    assert _row_button(cv, 0, cv._REMOVE_COL) is not None


def test_auto_frame_bounds_track_finite_curve_extent(qtbot):
    # A NaN-padded curve (the Impulse ∫p·dt trace is NaN before the onset)
    # must frame to where the curve actually exists, not the full sample axis --
    # otherwise Auto Frame stretches X across a sea of empty samples.
    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.graph import MetricGraph

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
    from sound_metric_app.ui.graph import MetricGraph

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
    from sound_metric_app.ui.graph import MetricGraph

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
    from sound_metric_app.ui.graph import MetricGraph

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
    assert graph._framing._frame_window_btn.isEnabled()
    assert graph._window_x_bounds == (1.0, 3.0)
    graph.frame_calc_window()
    view_x0, view_x1 = graph._plot.getViewBox().viewRange()[0]
    # Framed to the window, not the curve's [0.0, 4.0] extent; the small padding
    # keeps both lines off the very edge.
    assert 0.0 < view_x0 < 1.0
    assert 3.0 < view_x1 < 4.0
    # Y is unchanged from Auto Frame's: framing narrows X only.
    window_y = graph._plot.getViewBox().viewRange()[1]
    graph.auto_frame()
    assert graph._plot.getViewBox().viewRange()[1] == pytest.approx(window_y)

    # A trace with only one window edge -- or none at all -- has no span to
    # frame, so the button goes back to disabled.
    trace.window_end_index = None
    graph.show_trace(trace)
    assert not graph._framing._frame_window_btn.isEnabled()
    assert graph._window_x_bounds is None

    graph.show_message("nothing graphed")
    assert not graph._framing._frame_window_btn.isEnabled()


def test_frame_calc_window_autoranges_y_on_an_all_nan_curve(qtbot):
    # The window lines come from the trace's indices, which are set before the
    # finite-sample check -- so an all-NaN curve can still leave Frame Calc
    # Window enabled with no Y extent to pin to. Framing must fall back to
    # autoranging Y rather than unpacking a None _y_bounds.
    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.graph import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)

    trace = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        values=np.full(5, np.nan),
        y_label="Impulse ∫p·dt (Pa·ms)",
        title="Peak Impulse",
        window_start_index=1,
        window_end_index=3,
    )
    graph.show_trace(trace)
    assert graph._y_bounds is None
    assert graph._window_x_bounds == (1.0, 3.0)
    assert graph._framing._frame_window_btn.isEnabled()
    assert not graph._framing._auto_frame_btn.isEnabled()

    graph.frame_calc_window()  # must not raise
    view_x0, view_x1 = graph._plot.getViewBox().viewRange()[0]
    assert 0.0 < view_x0 < 1.0
    assert 3.0 < view_x1 < 4.0


def test_y_bounds_ignore_a_non_finite_level_line(qtbot):
    # trace.level is appended to a list already seeded with the curve's finite
    # min/max, so a NaN level compares False against the running accumulator and
    # drops out of min()/max() instead of poisoning the Y range.
    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.graph import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)

    trace = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0]),
        values=np.array([10.0, 20.0, 15.0]),
        y_label="SPL (dB)",
        title="Peak dB",
        level=float("nan"),
    )
    graph.show_trace(trace)
    assert graph._y_bounds == (10.0, 20.0)


def test_onset_zoom_frames_fixed_times_off_the_capture_axis(qtbot):
    # The onset close-ups frame *fixed* times -- a set start, a set width -- so
    # the same slice of every shot frames identically and close-ups compare shot
    # to shot. Nothing about the trace's own window moves them.
    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.graph import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)

    trace = MetricTrace(
        t_ms=np.arange(0.0, 100.0, 1.0),
        values=np.arange(0.0, 100.0, 1.0),
        y_label="SPL (dB)",
        title="Peak dB",
        window_start_index=20,
        window_end_index=None,
    )
    graph.show_trace(trace)
    # No end line, so Frame Calc Window is out -- but the close-ups need no
    # calculation window at all, only a drawn curve.
    assert not graph._framing._frame_window_btn.isEnabled()
    assert all(btn.isEnabled() for btn in graph._framing._frame_onset_btns)

    # One button per configured span, each labelled with and framing its own --
    # driven through the button, so a mis-bound click handler shows up here.
    assert [btn.text() for btn in graph._framing._frame_onset_btns] == [
        f"+{ms:g} ms" for ms in MetricGraph._ONSET_ZOOM_MS
    ]
    start = MetricGraph._ONSET_ZOOM_START_MS
    for btn, span in zip(graph._framing._frame_onset_btns, MetricGraph._ONSET_ZOOM_MS):
        btn.click()
        view_x0, view_x1 = graph._plot.getViewBox().viewRange()[0]
        # Anchored at the fixed start -- not the window start line at 20 ms --
        # with `span` ms to its right; the small padding keeps it off the edge.
        assert view_x0 == pytest.approx(start - span * 0.02)
        assert view_x1 == pytest.approx(start + span * 1.02)

    # Zooming in is a purely *horizontal* move: every framing button lands on
    # the same Y range Auto Frame does. Letting Y refit to the framed slice
    # rescaled the axis to a few dB and redrew the curve at a different height,
    # so the zoom levels could not be compared by eye.
    graph.auto_frame()
    auto_y = graph._plot.getViewBox().viewRange()[1]
    graph._framing._frame_onset_btns[0].click()
    assert graph._plot.getViewBox().viewRange()[1] == pytest.approx(auto_y)

    # A trace with no window at all still frames -- the times are the trace's,
    # not the window's -- but no curve at all leaves nothing to frame.
    trace.window_start_index = None
    graph.show_trace(trace)
    assert all(btn.isEnabled() for btn in graph._framing._frame_onset_btns)
    graph.show_message("nothing graphed")
    assert not any(btn.isEnabled() for btn in graph._framing._frame_onset_btns)


def _plain_trace(y_label: str = "SPL (dB)"):
    """A plain 0-4 ms / 1-3 dB trace, for the axis-bounds tests."""
    from sound_metric_app.dsp.graphing import MetricTrace

    return MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        values=np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        y_label=y_label,
        title="Peak dB",
    )


def _graph_with_a_curve(qtbot):
    """A graph showing one :func:`_plain_trace`."""
    from sound_metric_app.ui.graph import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)
    graph.show_trace(_plain_trace())
    return graph


def test_manual_axis_bounds_frame_exactly_and_outlive_the_framing_buttons(qtbot):
    # Typed bounds are the escape hatch from the framing buttons, so they must
    # land exactly where they were asked to -- no padding -- and a hand-set Y
    # must survive a later frame, which otherwise pins Y to the curve's extent.
    graph = _graph_with_a_curve(qtbot)
    assert graph._axis_bounds._button.isEnabled()

    graph.set_axis_bounds((1.5, 2.5), (0.0, 10.0))
    view_x, view_y = graph._plot.getViewBox().viewRange()
    assert view_x == pytest.approx([1.5, 2.5])
    assert view_y == pytest.approx([0.0, 10.0])

    # Framing stays a horizontal move -- but against the typed scale now, not
    # the curve's 1-3 dB extent.
    graph.auto_frame()
    view_x, view_y = graph._plot.getViewBox().viewRange()
    assert view_x == pytest.approx([0.0, 4.0])
    assert view_y == pytest.approx([0.0, 10.0])

    # A None range hands that axis back to autorange, and drops the override.
    graph.set_axis_bounds((1.5, 2.5), None)
    assert graph._manual_y_bounds is None
    graph.auto_frame()
    # Back to the curve's own extent (pyqtgraph pads it a little), not 0-10.
    y0, y1 = graph._plot.getViewBox().viewRange()[1]
    assert y0 < graph._y_bounds[0] and y1 > graph._y_bounds[1]
    assert y1 < 10.0

    # A framing click is a fresh X decision and drops the typed X (only X --
    # the button would otherwise appear to do nothing on the next redraw).
    graph.set_axis_bounds((1.5, 2.5), (0.0, 10.0))
    graph.auto_frame()
    assert graph._manual_x_bounds is None
    assert graph._manual_y_bounds == (0.0, 10.0)


def test_manual_axis_bounds_survive_a_redraw_of_the_same_metric(qtbot):
    # Hiding a curve, or adding a shot to an overlay, re-renders the same
    # metric -- and used to spring the frame back to autorange on every toggle.
    # A different quantity on the axes is a real reset: its numbers say nothing
    # about the scale that was chosen for this one.
    graph = _graph_with_a_curve(qtbot)
    graph.set_axis_bounds((1.5, 2.5), (0.0, 10.0))

    # Both views put "Loading…" up mid-redraw, so that must not undo them either.
    graph.show_message("Loading…")
    graph.show_trace(_plain_trace())
    view_x, view_y = graph._plot.getViewBox().viewRange()
    assert view_x == pytest.approx([1.5, 2.5])
    assert view_y == pytest.approx([0.0, 10.0])

    # An axis left automatic still refits to whatever was drawn. (pyqtgraph
    # defers an enabled autorange to the next paint, which never comes for an
    # unshown widget, so ask for it here.)
    graph.set_axis_bounds(None, (0.0, 10.0))
    graph.show_trace(_plain_trace())
    graph._plot.getViewBox().updateAutoRange()
    view_x, view_y = graph._plot.getViewBox().viewRange()
    assert view_x[0] < 0.0 and view_x[1] > 4.0
    assert view_y == pytest.approx([0.0, 10.0])

    # Switching metric drops them, and the plot goes back to autorange.
    graph.show_trace(_plain_trace(y_label="Impulse ∫p·dt (Pa·ms)"))
    assert graph._manual_y_bounds is None and graph._manual_x_bounds is None
    graph._plot.getViewBox().updateAutoRange()
    # Fitted to the curve's own 1-3 extent (padded a little), not the old 0-10.
    y0, y1 = graph._plot.getViewBox().viewRange()[1]
    assert y0 < 1.0 and 3.0 < y1 < 10.0


def test_axis_bounds_dialog_opens_on_the_current_view(qtbot, monkeypatch):
    # The form starts from what the operator is looking at (which is not the
    # trace's extent once they have panned), and applying it frames the plot.
    from PySide6 import QtWidgets

    from sound_metric_app.ui.graph.axis_bounds import AxisBoundsDialog

    graph = _graph_with_a_curve(qtbot)
    graph.set_axis_bounds((1.0, 3.0), (0.0, 8.0))

    opened: list = []

    def fake_exec(self):
        opened.append(
            [e.text() for e in (self.x_min_edit, self.x_max_edit,
                                self.y_min_edit, self.y_max_edit)]
        )
        self.y_max_edit.setText("9")
        self._on_accept()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(AxisBoundsDialog, "exec", fake_exec)
    graph.edit_axis_bounds()

    assert opened == [["1", "3", "0", "8"]]
    assert graph._plot.getViewBox().viewRange()[1] == pytest.approx([0.0, 9.0])

    # Nothing drawn: no view to edit, so the form never opens.
    graph.show_message("nothing graphed")
    graph.edit_axis_bounds()
    assert len(opened) == 1


def test_axis_bounds_dialog_validates_each_axis_as_a_pair(qtbot, monkeypatch):
    # Half a pair is ambiguous (pin or release?) and an inverted pair is not a
    # range; both are refused rather than guessed at. Clearing *both* boxes of
    # an axis is the one way to release it.
    from PySide6 import QtWidgets

    from sound_metric_app.ui.graph.axis_bounds import AxisBoundsDialog

    warned: list = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2])
    )

    def dialog():
        d = AxisBoundsDialog(x_range=(0.0, 4.0), y_range=(1.0, 3.0), y_label="dB")
        qtbot.addWidget(d)
        return d

    d = dialog()
    d.x_max_edit.clear()
    d._on_accept()
    assert warned and "both a min and a max for X" in warned[-1]
    assert d.result() != QtWidgets.QDialog.Accepted

    d = dialog()
    d.y_min_edit.setText("50")
    d._on_accept()
    assert "Y min must be below Y max" in warned[-1]

    d = dialog()
    d.x_min_edit.setText("not a number")
    d._on_accept()
    assert "is not a number" in warned[-1]

    # Reset empties every box but stays in the form, so releasing one axis and
    # retyping the other is a single trip through the dialog.
    d = dialog()
    d._on_reset()
    d.x_min_edit.setText("2")
    d.x_max_edit.setText("6")
    d._on_accept()
    assert d.result() == QtWidgets.QDialog.Accepted
    assert d.values() == ((2.0, 6.0), None)


def test_axis_bounds_dialog_prefill_survives_a_deep_zoom(qtbot):
    # The prefill is rounded for typeability, but never so far that it stops
    # describing the view: at a fixed 4 significant digits both ends of a
    # zoomed-in axis print the same string, and Apply on an untouched form
    # then trips the dialog's own low < high check.
    from PySide6 import QtWidgets

    from sound_metric_app.ui.graph.axis_bounds import AxisBoundsDialog

    d = AxisBoundsDialog(x_range=(10.52, 10.524), y_range=(163.451, 163.459))
    qtbot.addWidget(d)
    d._on_accept()
    assert d.result() == QtWidgets.QDialog.Accepted
    x_range, y_range = d.values()
    assert x_range == pytest.approx((10.52, 10.524))
    assert y_range == pytest.approx((163.451, 163.459))

    # …and re-framing on the prefilled numbers keeps the view where it was,
    # rather than nudging its edges to the nearest round value.
    d = AxisBoundsDialog(x_range=(10.4999, 15.5001), y_range=(0.0, 8.0))
    qtbot.addWidget(d)
    d._on_accept()
    assert d.values()[0] == (10.4999, 15.5001)

    # Float noise still collapses: that is what the rounding is for.
    d = AxisBoundsDialog(x_range=(0.0, 2.9999999999996), y_range=(0.0, 8.0))
    qtbot.addWidget(d)
    assert d.x_max_edit.text() == "3"


def test_window_marker_labels_run_vertically_from_the_top(qtbot):
    # The labels are rotated parallel to their line and top-aligned. Guards the
    # anchor choice: pyqtgraph's default anchors for rotated text centre the
    # label on `position`, which with position~1.0 hangs half of it above the
    # view and clips it. Anchoring the far end pins the top edge instead.
    import pyqtgraph as pg

    from sound_metric_app.dsp.graphing import MetricTrace
    from sound_metric_app.ui.graph import MetricGraph

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
    from sound_metric_app.ui.format import _unit_of
    from sound_metric_app.ui.graph import MetricGraph

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
    assert not graph._readout._readout_label.isVisible()
    assert graph._readout._pick_marker is None

    # Picking a sample fills the box with its value + unit + time and marks it.
    graph._show_readout(1, 142.5)
    assert graph._readout._pick_marker is not None
    text = graph._readout._readout_label.text()
    assert "142.500" in text and "dBA" in text and "1.00 ms" in text

    # Clear removes the marker and hides the box.
    graph.clear_readout()
    assert graph._readout._pick_marker is None
    assert not graph._readout._readout_label.isVisible()

    # Drawing a fresh trace also drops any prior pick.
    graph._show_readout(0, 100.0)
    graph.show_trace(trace)
    assert graph._readout._pick_marker is None
    assert not graph._readout._readout_label.isVisible()


#: Two palette entries, named so the overlay tests read as "these two colours".
BLUE = (66, 135, 245)
RED = (214, 90, 70)


def _overlay_traces():
    """Two curves of the same metric with deliberately different extents.

    A runs 0-2 ms and is all-finite; B is NaN until 2 ms and runs to 4 ms, and
    their calculation windows only partly overlap — so every union the overlay
    takes is a different number from either curve's own.
    """
    from sound_metric_app.dsp.graphing import MetricTrace

    a = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 5.0, 2.0]),
        y_label="Impulse ∫p·dt (Pa·ms)",
        title="Peak Impulse",
        peak_index=1,
        connected=True,
        window_start_index=0,
        window_end_index=1,
    )
    b = MetricTrace(
        t_ms=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        values=np.array([np.nan, np.nan, 3.0, 9.0, 4.0]),
        y_label="Impulse ∫p·dt (Pa·ms)",
        title="Peak Impulse",
        peak_index=3,
        connected=True,
        window_start_index=1,
        window_end_index=4,
    )
    return a, b


def test_overlaid_series_draw_with_a_legend_and_union_bounds(qtbot):
    # The Compare tab hands the same widget several traces instead of one. Each
    # gets its own colour and legend row, and every bound the framing buttons
    # use widens to cover all of them -- a bound fitted to whichever curve was
    # drawn first would frame away the others.
    from sound_metric_app.ui.graph import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)
    a, b = _overlay_traces()

    graph.show_traces([("first", a, BLUE), ("second", b, RED)], "two shots")
    assert len(graph._plot.listDataItems()) == 2
    assert graph._legend.isVisible() and len(graph._legend.items) == 2
    # X spans A's start to B's end; Y spans A's floor to B's ceiling; the window
    # brackets the earliest start and the latest end of the two.
    assert graph._x_bounds == (0.0, 4.0)
    assert graph._y_bounds == (1.0, 9.0)
    assert graph._window_x_bounds == (0.0, 4.0)
    assert graph._framing._frame_window_btn.isEnabled()
    assert graph._framing._auto_frame_btn.isEnabled()

    # Framing still lands on those unions, so both curves stay in view.
    graph.auto_frame()
    view_x0, view_x1 = graph._plot.getViewBox().viewRange()[0]
    assert (view_x0, view_x1) == pytest.approx((0.0, 4.0))

    # Each curve is drawn in the colour it was handed, so a caller that can hide
    # one keeps the rest on the colours they already had.
    pens = [item.opts["pen"].color().getRgb()[:3] for item in graph._plot.listDataItems()]
    assert pens == [BLUE, RED]
    # The palette itself lives here, so a view listing the same series reads it
    # from the graph rather than keeping a second copy.
    assert MetricGraph.series_color(0) == MetricGraph._SERIES_COLORS[0]
    assert MetricGraph.series_color(len(MetricGraph._SERIES_COLORS)) == MetricGraph.series_color(0)


def test_single_trace_keeps_the_plain_legend_free_graph(qtbot):
    # The Batch average tab draws one curve through the same code path. It must
    # come out exactly as before: no legend, and the bounds of that one trace.
    from sound_metric_app.ui.graph import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)
    a, b = _overlay_traces()

    graph.show_traces([("first", a, BLUE), ("second", b, RED)], "two shots")
    graph.show_trace(a)
    assert len(graph._plot.listDataItems()) == 1
    assert not graph._legend.isVisible()  # and the stale two rows are gone
    assert len(graph._legend.items) == 0
    assert graph._x_bounds == (0.0, 2.0)
    assert graph._window_x_bounds == (0.0, 1.0)

    graph.show_message("nothing graphed")
    assert not graph._legend.isVisible()


def test_readout_names_which_overlaid_curve_was_picked(qtbot):
    # With one curve the number speaks for itself; with several it does not say
    # which shot it came from, so the series' label leads the readout.
    from sound_metric_app.ui.graph import MetricGraph

    graph = MetricGraph()
    qtbot.addWidget(graph)
    a, b = _overlay_traces()

    graph.show_trace(a)
    graph._show_readout(1, 5.0)
    assert graph._readout._readout_label.text().startswith("5.000")

    graph.show_traces([("first", a, BLUE), ("second", b, RED)], "two shots")
    graph._show_readout(3, 9.0, series_index=1)
    assert graph._readout._readout_label.text().startswith("second:  9.000")


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


def test_data_bank_tree_opens_collapsed_and_keeps_what_the_user_opened(window, qtbot):
    """Collapsed on arrival, but a refresh must not undo the user's expanding.

    Every edit, tick and bring-forward rebuilds this tree from scratch. If the
    rebuild always collapsed, the user would be thrown back to the top of the
    archive after each click.
    """
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    combination_item, batch_item, cluster_item, _shot = _tree_nodes(bv)
    assert not combination_item.isExpanded()
    assert not batch_item.isExpanded()

    # Open a branch down to the shots, then force the rebuild an edit would.
    combination_item.setExpanded(True)
    batch_item.setExpanded(True)
    bv.refresh()

    # Fresh items -- the old ones were freed by the rebuild -- carrying the old
    # state, matched by record identity rather than row position.
    combination_item, batch_item, cluster_item, _shot = _tree_nodes(bv)
    assert combination_item.isExpanded()
    assert batch_item.isExpanded()
    assert not cluster_item.isExpanded()  # never opened: still closed


def test_report_tree_opens_collapsed_and_keeps_open_slots(window, qtbot):
    _mark_all_shots(window, qtbot)
    _include_everything(window)

    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.tree.topLevelItemCount() == 4, timeout=5000)
    slot = next(
        rv.tree.topLevelItem(i)
        for i in range(rv.tree.topLevelItemCount())
        if rv.tree.topLevelItem(i).childCount()
    )
    assert not slot.isExpanded()
    label = slot.text(0)

    slot.setExpanded(True)
    rv.refresh()
    reopened = next(
        rv.tree.topLevelItem(i)
        for i in range(rv.tree.topLevelItemCount())
        if rv.tree.topLevelItem(i).text(0) == label
    )
    assert reopened.isExpanded()


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


def test_data_bank_compare_buttons_pin_an_idle_shots_ml_and_se(window, qtbot):
    # The data bank is the one tab that can reach a shot before it is ever
    # brought forward -- that idle reach is the whole point of pinning from
    # here rather than only from Batch average.
    from sound_metric_app.models import MicPosition
    from sound_metric_app.ui.compare_series import _COMPARE_ADDED_LABEL

    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)
    assert shot.included is False  # never brought forward

    widget = bv.tree.itemWidget(shot_item, bv._COMPARE_COL)
    buttons = widget.findChildren(QtWidgets.QPushButton)
    assert [b.text() for b in buttons] == ["ML", "SE"]

    cv = window.compare_view
    for button in buttons:
        button.click()
        assert button.text() == _COMPARE_ADDED_LABEL

    assert {s.position for s in cv._series} == {MicPosition.ML, MicPosition.SE}
    assert {s.shot_id for s in cv._series} == {shot.id}
    # The idle status is surfaced in the tooltip/detail rather than hidden.
    assert all("idle" in s.detail for s in cv._series)


def test_data_bank_shot_row_shows_both_mics_pretrigger_floors(window, qtbot):
    # The diagnostic has to be readable from the data bank, which is where an
    # operator scans the whole archive — including the idle shots the Batch
    # average tab never shows. A shot row stands for the capture, so it carries
    # both mics in the one cell.
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _, _, cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)

    with window.controller._repo() as repo:
        repo._conn.execute(
            "UPDATE channel_metrics SET pretrigger_floor_pa = ? "
            "WHERE shot_id = ? AND mic_position = 'ML'",
            (0.612, shot.id),
        )
        repo._conn.execute(
            "UPDATE channel_metrics SET pretrigger_floor_pa = ? "
            "WHERE shot_id = ? AND mic_position = 'SE'",
            (-0.25, shot.id),
        )
        repo._conn.commit()
    bv.refresh()
    _, _, cluster_item, shot_item = _tree_nodes(bv)

    # Signed, three decimals, both positions in fixed order so a column of rows
    # can be scanned straight down.
    assert shot_item.text(bv._FLOOR_COL) == "ML:+0.612  SE:-0.250"
    # Container rows have no single baseline, so they stay blank rather than
    # showing an aggregate that would hide the one bad capture.
    assert cluster_item.text(bv._FLOOR_COL) == ""


def test_data_bank_shot_row_marks_an_unmeasured_floor_rather_than_showing_zero(
    window, qtbot
):
    # A row written before the column existed holds NULL. It must read as "not
    # measured", never as a healthy 0.000 baseline — that distinction is the
    # whole reason the repository omits NULLs instead of defaulting them.
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)

    with window.controller._repo() as repo:
        repo._conn.execute(
            "UPDATE channel_metrics SET pretrigger_floor_pa = NULL WHERE shot_id = ?",
            (shot.id,),
        )
        repo._conn.commit()
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)

    assert shot_item.text(bv._FLOOR_COL) == "—"


def test_data_bank_compare_button_only_appears_for_a_marked_channel(window, qtbot):
    # A shot row's Compare cell has one button per tagged channel -- not scoped
    # to a single mic the way a Batch average shot row is (it sits under one
    # position's slot). A single-mic shot only gets one.
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    _kind, shot, *_rest = shot_item.data(0, QtCore.Qt.UserRole)
    with window.controller._repo() as repo:
        repo._conn.execute("UPDATE shots SET se_channel = NULL WHERE id = ?", (shot.id,))
        repo._conn.commit()
    bv.refresh()
    _, _, _cluster_item, shot_item = _tree_nodes(bv)

    widget = bv.tree.itemWidget(shot_item, bv._COMPARE_COL)
    buttons = widget.findChildren(QtWidgets.QPushButton)
    assert [b.text() for b in buttons] == ["ML"]


def test_pinning_the_same_shot_from_data_bank_and_batch_average_is_a_no_op(window, qtbot):
    # The user's worry: picking the same shot/mic from two different tabs must
    # not draw it twice. CompareSeries keys on (shot_id, position) regardless
    # of which tab built it, so the second pin -- from whichever tab it comes
    # from -- has to be refused exactly like a same-tab repeat.
    from sound_metric_app.ui.compare_series import (
        _COMPARE_ADDED_LABEL,
        _COMPARE_ALREADY_LABEL,
    )

    rv = _loaded_report(window, qtbot)  # marks + includes every shot
    bv = window.bank_view
    bv.refresh()
    cv = window.compare_view

    report_button = _frp_compare_buttons(rv)[0]  # an ML or SE FRP shot row
    report_button.click()
    assert report_button.text() == _COMPARE_ADDED_LABEL
    assert len(cv._series) == 1
    pinned_key = cv._series[0].key

    # Grab both buttons by their stable ML/SE identity *before* clicking either
    # -- a click flashes the label to "Added"/"Pinned", so matching by current
    # text after that point would grab the wrong one.
    _, _, _cluster_item, shot_item = _tree_nodes(bv)
    bank_widget = bv.tree.itemWidget(shot_item, bv._COMPARE_COL)
    bank_buttons = {b.text(): b for b in bank_widget.findChildren(QtWidgets.QPushButton)}
    bank_button = bank_buttons[pinned_key[1].value]
    other_bank_button = bank_buttons[next(v for v in bank_buttons if v != pinned_key[1].value)]

    bank_button.click()
    # Same (shot_id, position): refused, not duplicated, and the button says so.
    assert bank_button.text() == _COMPARE_ALREADY_LABEL
    assert len(cv._series) == 1

    # And the reverse direction: pin from the data bank first, then repeat from
    # Batch average for the shot/mic that button was never scoped to.
    other_bank_button.click()
    assert other_bank_button.text() == _COMPARE_ADDED_LABEL
    assert len(cv._series) == 2

    other_report_button = next(
        b for b in _frp_compare_buttons(rv) if b is not report_button
    )
    other_report_button.click()
    assert other_report_button.text() == _COMPARE_ALREADY_LABEL
    assert len(cv._series) == 2


def test_edit_batch_session_metadata_via_tree(window, qtbot, monkeypatch):
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    _combination_item, batch_item, *_rest = _tree_nodes(bv)
    bv.tree.setCurrentItem(batch_item)

    from PySide6 import QtWidgets

    from sound_metric_app.ui.dialogs import BatchEditDialog

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
    from sound_metric_app.ui.dialogs import BatchEditDialog

    warned: list = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))

    dialog = BatchEditDialog(Batch(id=1, combination_id=1), combination_label="X", parent=window)
    dialog.date_edit.setText("22-07-2026")
    dialog._on_accept()

    assert warned and "YYYY-MM-DD" in warned[0]
    assert dialog.result() != QtWidgets.QDialog.Accepted


def test_double_click_edits_leaf_and_batch_rows_only(window, qtbot, monkeypatch):
    # Only editable rows (batch, shot) route through _edit_selected; a pure
    # container never pops the modal.
    _mark_all_shots(window, qtbot)

    bv = window.bank_view
    bv.refresh()
    edited: list = []
    monkeypatch.setattr(bv, "_edit_selected", lambda: edited.append(True))

    # A batch row is editable *and* has children, so Qt's expand/collapse must
    # not ride along on the same double-click that opens the edit modal.
    assert not bv.tree.expandsOnDoubleClick()

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
    from sound_metric_app.ui.dialogs import ShotEditDialog

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
    from sound_metric_app.ui.dialogs import ShotEditDialog

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


def test_mark_form_rejects_a_below_one_cluster_before_dsp(window, qtbot, monkeypatch):
    # The Mark tab's Cluster override shares the edit dialog's 1-based
    # constraint, so an explicit 0 must be caught up front — not deferred to the
    # service on the worker thread after the capture is read and the DSP has run.
    from PySide6 import QtWidgets

    window.ingest_view._ingest()
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 2, timeout=5000)
    first_id = int(window.ingest_view.table.item(0, 0).text())
    window.open_marking_for(first_id)
    mv = window.marking_view
    qtbot.waitUntil(lambda: mv.ml_combo.isEnabled() and mv.ml_combo.count() >= 3, timeout=5000)

    warned: list = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    dispatched: list = []
    monkeypatch.setattr(mv, "_run_async", lambda *a, **k: dispatched.append(a))

    mv.ammo_combo.setCurrentText("M855")
    mv.cluster_edit.setText("0")
    mv._mark()

    # Immediate front-end rejection with the same message the edit dialog gives,
    # and no marking work dispatched.
    assert warned == ["A cluster of 1 or greater is required."]
    assert dispatched == []

    # A blank cluster stays valid — it falls back to the filename cluster — so
    # marking proceeds to the async worker.
    warned.clear()
    mv.cluster_edit.clear()
    mv._mark()
    assert warned == []
    assert dispatched


def test_shot_edit_dialog_rejects_a_bad_mic_tagging(window, monkeypatch):
    # The mark form and the edit dialog share one tagging check, so the two
    # warnings are asserted here rather than duplicated per caller.
    from PySide6 import QtWidgets

    from sound_metric_app.models import Shot
    from sound_metric_app.ui.dialogs import ShotEditDialog

    warned: list = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[1]))

    dialog = ShotEditDialog(
        Shot(source_file="f.dxd", shot_order=1, ml_channel="AI 1"),
        sku="SUP-1",
        platform="AR15",
        ammo="M855",
        cluster_index=1,
        channel_names=["AI 1", "AI 2"],
        parent=window,
    )

    # Both mics on one channel would attribute one waveform to two positions.
    dialog.se_combo.setCurrentText("AI 1")
    dialog._on_accept()
    assert warned == ["Same channel"]

    # Nothing tagged leaves no channel to compute metrics from.
    dialog.ml_combo.setCurrentIndex(0)
    dialog.se_combo.setCurrentIndex(0)
    dialog._on_accept()
    assert warned == ["Same channel", "No mic tagged"]
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
    from sound_metric_app.ui.dialogs import AmmoDefinitionsDialog

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
