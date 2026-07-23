"""Tests for the `sma` workflow CLI (BUILD_PLAN Task 8).

The channel/capture readers are patched at module scope so no real ``.dxd`` file
is needed; ``mark`` still runs the *real* DSP over synthetic sine frames, so the
full ingest -> mark -> include -> report cycle is exercised end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from sound_metric_app import workflow_cli
from sound_metric_app.models import Frame
from sound_metric_app.storage import WorkflowRepository

FS = 200_000.0


def _sine_frame(path: str, channel: str, freq: float = 1000.0, amp: float = 1.0) -> Frame:
    t = np.arange(20_000) / FS
    return Frame(
        samples=amp * np.sin(2 * np.pi * freq * t),
        sample_rate=FS,
        channel=channel,
        source_file=path,
        timestamp=None,
    )


@pytest.fixture
def capture_reader(monkeypatch):
    """Patch the CLI's capture reader to yield the two DAQ mic channels."""

    def reader(path: str) -> list[Frame]:
        return [_sine_frame(path, "AI 1"), _sine_frame(path, "AI 2")]

    monkeypatch.setattr(workflow_cli, "_capture_reader", reader)
    return reader


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated settings file + a tmp DB path; returns (db_path, inbox)."""
    monkeypatch.setenv("SMA_CONFIG", str(tmp_path / "sma_config.json"))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return str(tmp_path / "wf.db"), inbox


def _touch(folder, name):
    (folder / name).write_bytes(b"")


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #


def test_full_ingest_mark_include_report_cycle(env, capture_reader, capsys):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_0000.dxd")  # Dewesoft's 0-based counter: the FRP
    _touch(inbox, "SUP-1_AR15_01_0001.dxd")

    # Configure the input folder, then ingest from it (no folder arg).
    assert workflow_cli.main(["config", "set-input-folder", str(inbox)]) == 0
    assert workflow_cli.main(["ingest", "--db", db, "--no-validate"]) == 0
    out = capsys.readouterr().out
    assert "ingested       : 2" in out
    assert "[cluster 1, shot 0]" in out

    # Both shots surface as unmarked.
    assert workflow_cli.main(["list", "unmarked", "--db", db]) == 0
    assert "Unmarked shots (2)" in capsys.readouterr().out

    # Mark both. No --se/--ml needed: AI 1 / AI 2 auto-tag.
    for shot_id in (1, 2):
        assert workflow_cli.main(["mark", str(shot_id), "--ammo", "M855", "--db", db]) == 0
    mark_out = capsys.readouterr().out
    assert "Marked shot #2" in mark_out
    assert "SE:" in mark_out and "ML:" in mark_out
    assert "Cluster 1" in mark_out
    assert "(FRP)" in mark_out and "(Regular)" in mark_out
    # Marking files a shot in the data bank idle, not into the average.
    assert "idle" in mark_out

    # Nothing left unmarked.
    assert workflow_cli.main(["list", "unmarked", "--db", db]) == 0
    assert "No unmarked shots." in capsys.readouterr().out

    # Before anything is brought forward the average is empty.
    assert workflow_cli.main(["report", "--batch", "1", "--db", db]) == 0
    assert "nothing brought forward yet" in capsys.readouterr().out

    # The data bank still lists both shots, unticked.
    assert workflow_cli.main(["bank", "1", "--db", db]) == 0
    bank = capsys.readouterr().out
    assert "[ ] #1" in bank and "[ ] #2" in bank
    assert "FRP: 0/3   Regular: 0/5" in bank

    # Bring the whole cluster forward, then report the four slots.
    assert workflow_cli.main(["include", "cluster", "1", "--db", db]) == 0
    assert "FRP: 1/3   Regular: 1/5" in capsys.readouterr().out
    assert workflow_cli.main(["report", "--batch", "1", "--db", db]) == 0
    report = capsys.readouterr().out
    assert "SUP-1 / AR15 / M855" in report
    assert "Muzzle Left · FRP" in report and "Shooter's Ear · Regular" in report
    assert "2 of 2 shot(s) included" in report

    # Close the batch; it then lists as closed.
    assert workflow_cli.main(["close-batch", "1", "--db", db]) == 0
    assert "Closed batch #1" in capsys.readouterr().out
    assert workflow_cli.main(["list", "batches", "--db", db]) == 0
    assert "[closed]" in capsys.readouterr().out


def test_report_does_not_crash_on_blanked_metric_averages(env, capture_reader, capsys):
    # After a migration blanks a legacy database's metric columns, a report run
    # before the shots are re-marked must degrade to "—", not crash on
    # f"{None:.2f}". The slot's averages dict is non-empty (positions present),
    # so the "nothing brought forward" guard does not catch this.
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    workflow_cli.main(["mark", "1", "--ammo", "M855", "--db", db])
    workflow_cli.main(["include", "shot", "1", "--db", db])
    capsys.readouterr()

    # Simulate the blanked (post-migration, pre-re-mark) state.
    with WorkflowRepository(db) as repo:
        cols = ", ".join(
            f"{c} = NULL"
            for c in (
                "peak_pa", "peak_db", "peak_a_pa", "peak_dba", "impulse_pa_ms",
                "peak_impulse_db", "leq10ms_pa", "leq10ms_db", "liaeq_pa", "liaeq_100ms_db",
            )
        )
        repo._conn.execute(f"UPDATE channel_metrics SET {cols}")
        repo._conn.commit()
        combination_id = repo.all_combinations()[0].id

    # Both the --batch and --combination report paths must survive.
    assert workflow_cli.main(["report", "--batch", "1", "--db", db]) == 0
    batch_out = capsys.readouterr().out
    assert "(n=1)" in batch_out and "—" in batch_out

    assert workflow_cli.main(["report", "--combination", str(combination_id), "--db", db]) == 0
    assert "—" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


def test_ingest_rescan_adds_zero(env, capsys):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    capsys.readouterr()
    assert workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"]) == 0
    out = capsys.readouterr().out
    assert "ingested       : 0" in out
    assert "already present : 1" in out


def test_ingest_without_folder_or_config_errors(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["ingest", "--db", db, "--no-validate"])
    assert rc == 2
    assert "No input folder" in capsys.readouterr().err


def test_ingest_reports_malformed(env, capsys):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    _touch(inbox, "not_enough.dxd")
    _touch(inbox, "SUP-1_AR15_001.dxd")  # the old 3-field convention
    assert workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"]) == 0
    out = capsys.readouterr().out
    assert "ingested       : 1" in out
    assert "malformed       : 2" in out


# --------------------------------------------------------------------------- #
# mark
# --------------------------------------------------------------------------- #


def test_mark_auto_tags_without_channel_flags(env, capsys, capture_reader):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    capsys.readouterr()
    assert workflow_cli.main(["mark", "1", "--ammo", "M855", "--db", db]) == 0
    with WorkflowRepository(db) as repo:
        shot = repo.get_shot(1)
    assert (shot.ml_channel, shot.se_channel) == ("AI 1", "AI 2")


def test_mark_rejects_identical_se_and_ml(env, capsys, capture_reader):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    capsys.readouterr()
    rc = workflow_cli.main(
        ["mark", "1", "--ammo", "M855", "--se", "AI 1", "--ml", "AI 1", "--db", db]
    )
    assert rc == 2
    assert "cannot be the same channel" in capsys.readouterr().err


def test_mark_unknown_shot_errors(env, capsys, capture_reader):
    db, _ = env
    rc = workflow_cli.main(["mark", "999", "--ammo", "M855", "--db", db])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_mark_override_keys(env, capsys, capture_reader):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    capsys.readouterr()
    rc = workflow_cli.main(
        ["mark", "1", "--ammo", "M855", "--sku", "SUP-9", "--cluster", "4", "--db", db]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUP-9 / AR15 / M855" in out
    assert "Cluster 4" in out


# --------------------------------------------------------------------------- #
# data bank & inclusion
# --------------------------------------------------------------------------- #


@pytest.fixture
def marked_batch(env, capture_reader, capsys):
    """A batch of two clusters (3 + 4 shots), all marked and idle."""
    db, inbox = env
    for cluster, n in ((1, 3), (2, 4)):
        for order in range(n):  # 0-based: each cluster's 0000 is its FRP
            _touch(inbox, f"SUP-1_AR15_{cluster:02d}_{order:04d}.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    for shot_id in range(1, 8):
        workflow_cli.main(["mark", str(shot_id), "--ammo", "M855", "--db", db])
    capsys.readouterr()
    return db


def test_bank_lists_every_cluster_and_shot(marked_batch, capsys):
    assert workflow_cli.main(["bank", "1", "--db", marked_batch]) == 0
    out = capsys.readouterr().out
    assert "Cluster 1" in out and "Cluster 2" in out
    # The complete archive: all seven shots, none hidden for being idle.
    assert out.count("[ ] #") == 7
    assert "FRP       " in out and "Regular   " in out


def test_include_and_exclude_move_shots_between_the_views(marked_batch, capsys):
    db = marked_batch
    # Whole clusters cannot land on exactly 5 regulars, so mix the two grains:
    # cluster 1 wholesale (1 FRP + 2 regulars), then cluster 2's regulars.
    assert workflow_cli.main(["include", "cluster", "1", "--db", db]) == 0
    assert "FRP: 1/3   Regular: 2/5" in capsys.readouterr().out
    for shot_id in (5, 6, 7):
        workflow_cli.main(["include", "shot", str(shot_id), "--db", db])
    assert "FRP: 1/3   Regular: 5/5" in capsys.readouterr().out

    # An exclusion records why, and the reason shows in the data bank.
    assert workflow_cli.main(
        ["exclude", "shot", "7", "--reason", "high winds", "--db", db]
    ) == 0
    capsys.readouterr()
    assert workflow_cli.main(["bank", "1", "--db", db]) == 0
    bank = capsys.readouterr().out
    assert "high winds" in bank
    assert "FRP: 1/3   Regular: 4/5" in bank
    # Nothing was deleted: the archive still holds all seven shots.
    assert bank.count("#") >= 7


def test_include_unknown_shot_errors(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["include", "shot", "42", "--db", db])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# batch session metadata
# --------------------------------------------------------------------------- #


def test_batch_sets_session_metadata(marked_batch, capsys):
    db = marked_batch
    rc = workflow_cli.main(
        [
            "batch", "1", "--label", "Morning string", "--date", "2026-07-22",
            "--wind-speed", "4", "--temp", "88", "--rh", "35",
            "--notes", "clear, light crosswind", "--db", db,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Morning string 2026-07-22" in out
    assert "wind 4 mph" in out and "88 °F" in out and "RH 35%" in out
    assert "clear, light crosswind" in out


def test_batch_edits_one_field_and_keeps_the_rest(marked_batch, capsys):
    """An absent flag means "unchanged" on the CLI, unlike the full-form GUI write."""
    db = marked_batch
    assert workflow_cli.main(
        [
            "batch", "1", "--label", "Morning string", "--date", "2026-07-20",
            "--wind-speed", "5", "--temp", "88", "--rh", "35",
            "--notes", "calm", "--db", db,
        ]
    ) == 0
    capsys.readouterr()

    assert workflow_cli.main(["batch", "1", "--notes", "gusty by noon", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "gusty by noon" in out
    assert "Morning string 2026-07-20" in out
    assert "wind 5 mph" in out and "88 °F" in out and "RH 35%" in out

    with WorkflowRepository(db) as repo:
        batch = repo.get_batch(1)
    assert batch.label == "Morning string"
    assert batch.session_date == "2026-07-20"
    assert (batch.wind_speed, batch.temp, batch.relative_humidity) == (5.0, 88.0, 35.0)
    assert batch.notes == "gusty by noon"


def test_batch_clear_blanks_named_fields_only(marked_batch, capsys):
    db = marked_batch
    assert workflow_cli.main(
        ["batch", "1", "--label", "Morning string", "--wind-speed", "5", "--db", db]
    ) == 0
    capsys.readouterr()

    assert workflow_cli.main(
        ["batch", "1", "--clear", "wind-speed", "--notes", "windy", "--db", db]
    ) == 0
    assert "(none)" in capsys.readouterr().out

    with WorkflowRepository(db) as repo:
        batch = repo.get_batch(1)
    assert batch.wind_speed is None
    assert batch.label == "Morning string"
    assert batch.notes == "windy"


def test_batch_clear_conflicting_with_a_value_errors(marked_batch, capsys):
    db = marked_batch
    rc = workflow_cli.main(["batch", "1", "--clear", "notes", "--notes", "x", "--db", db])
    assert rc == 2
    assert "contradict" in capsys.readouterr().err


def test_batch_unknown_id_errors(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["batch", "42", "--label", "nope", "--db", db])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# list / report edge cases
# --------------------------------------------------------------------------- #


def test_list_clusters_requires_batch(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["list", "clusters", "--db", db])
    assert rc == 2
    assert "requires --batch" in capsys.readouterr().err


def test_list_clusters_unknown_batch_errors(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["list", "clusters", "--batch", "42", "--db", db])
    assert rc == 2
    assert "No batch with id 42" in capsys.readouterr().err


def test_list_clusters_shows_inclusion_counts(marked_batch, capsys):
    db = marked_batch
    workflow_cli.main(["include", "cluster", "1", "--db", db])
    capsys.readouterr()
    assert workflow_cli.main(["list", "clusters", "--batch", "1", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "(3 shot(s), 3 included)" in out
    assert "(4 shot(s), 0 included)" in out


def test_list_combinations(marked_batch, capsys):
    assert workflow_cli.main(["list", "combinations", "--db", marked_batch]) == 0
    out = capsys.readouterr().out
    assert "SUP-1 / AR15 / M855" in out
    assert "1 batch(es)" in out


def test_report_unknown_batch_errors(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["report", "--batch", "42", "--db", db])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_close_unknown_batch_errors(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["close-batch", "42", "--db", db])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_report_shows_empty_slots_as_none_included(marked_batch, capsys):
    # Only the FRP of cluster 1 is brought forward, so the regular slots are
    # visibly empty rather than silently absent.
    db = marked_batch
    workflow_cli.main(["include", "shot", "1", "--db", db])
    capsys.readouterr()
    assert workflow_cli.main(["report", "--batch", "1", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "Muzzle Left · FRP" in out
    assert "Muzzle Left · Regular      (none included)" in out


def test_report_combination_target(env, capsys, capture_reader):
    """`report --combination` walks every session under one test combination."""
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    workflow_cli.main(["mark", "1", "--ammo", "M855", "--db", db])
    capsys.readouterr()

    with WorkflowRepository(db) as repo:
        combination_id = repo.all_combinations()[0].id
    assert workflow_cli.main(["report", "--combination", str(combination_id), "--db", db]) == 0
    assert "SUP-1 / AR15 / M855" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_show_reports_unset_then_set(env, capsys):
    db, inbox = env
    workflow_cli.main(["config", "show"])
    out = capsys.readouterr().out
    assert "(unset)" in out
    # The soft targets are part of the visible configuration.
    assert "FRP 3, regular 5" in out

    workflow_cli.main(["config", "set-input-folder", str(inbox)])
    capsys.readouterr()
    workflow_cli.main(["config", "show"])
    out = capsys.readouterr().out
    assert str(inbox.resolve()) in out
