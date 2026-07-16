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
