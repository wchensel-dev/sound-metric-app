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
        mv.ammo_edit.setText("M855")  # SE/MR default to AI 1 / AI 2
        mv._mark()
        qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() < 2, timeout=5000)
        # loop condition re-reads the table; wait for the second mark to clear it
    qtbot.waitUntil(lambda: window.ingest_view.table.rowCount() == 0, timeout=5000)

    # --- Report shows the group with SE and MR rows (never mixed) ---
    rv = window.report_view
    rv.refresh()
    qtbot.waitUntil(lambda: rv.table.rowCount() >= 2, timeout=5000)
    mics = {rv.table.item(r, 1).text() for r in range(rv.table.rowCount())}
    assert {"SE", "MR"} <= mics

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
        mv.ammo_edit.setText("M855")
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
    mv.ammo_edit.setText("M855")
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
    mv.ammo_edit.setText("M855")
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
    mv.ammo_edit.setText("WRONG")
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
    assert dialog.ammo_edit.text() == "WRONG"  # pre-filled from the group
    assert dialog.se_combo.currentText() == "AI 1"  # pre-filled from the shot tags
    dialog.ammo_edit.setText("M855")
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
