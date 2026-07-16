"""Tests for the `sma` workflow CLI (BUILD_PLAN Task 8).

The channel/capture readers are patched at module scope so no real ``.dxd`` file
is needed; ``mark`` still runs the *real* DSP over synthetic sine frames, so the
full ingest -> mark -> close -> report cycle is exercised end to end.
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
    """Patch the CLI's capture reader to yield two synthetic mic channels."""

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


def test_full_ingest_mark_close_report_cycle(env, capture_reader, capsys):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_001.dxd")
    _touch(inbox, "SUP-1_AR15_002.dxd")

    # Configure the input folder, then ingest from it (no folder arg).
    assert workflow_cli.main(["config", "set-input-folder", str(inbox)]) == 0
    assert workflow_cli.main(["ingest", "--db", db, "--no-validate"]) == 0
    out = capsys.readouterr().out
    assert "ingested       : 2" in out

    # Both shots surface as unmarked.
    assert workflow_cli.main(["list", "unmarked", "--db", db]) == 0
    assert "Unmarked shots (2)" in capsys.readouterr().out

    # Mark both with SE + MR tags.
    for shot_id in (1, 2):
        rc = workflow_cli.main(
            ["mark", str(shot_id), "--ammo", "M855", "--se", "AI 1", "--mr", "AI 2", "--db", db]
        )
        assert rc == 0
    mark_out = capsys.readouterr().out
    assert "Marked shot #2" in mark_out
    assert "SE:" in mark_out and "MR:" in mark_out

    # Nothing left unmarked.
    assert workflow_cli.main(["list", "unmarked", "--db", db]) == 0
    assert "No unmarked shots." in capsys.readouterr().out

    # Report shows one group with SE and MR averages.
    assert workflow_cli.main(["report", "--batch", "1", "--db", db]) == 0
    report = capsys.readouterr().out
    assert "AR15 / M855" in report
    assert "SE (n=2)" in report and "MR (n=2)" in report

    # Close the batch; it then lists as closed.
    assert workflow_cli.main(["close-batch", "1", "--db", db]) == 0
    assert "Closed batch #1" in capsys.readouterr().out
    assert workflow_cli.main(["list", "batches", "--db", db]) == 0
    assert "[closed]" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


def test_ingest_rescan_adds_zero(env, capsys):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_001.dxd")
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
    _touch(inbox, "SUP-1_AR15_001.dxd")
    _touch(inbox, "not_enough.dxd")
    assert workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"]) == 0
    out = capsys.readouterr().out
    assert "ingested       : 1" in out
    assert "malformed       : 1" in out


# --------------------------------------------------------------------------- #
# mark
# --------------------------------------------------------------------------- #


def test_mark_requires_a_channel_tag(env, capsys):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    capsys.readouterr()
    rc = workflow_cli.main(["mark", "1", "--ammo", "M855", "--db", db])
    assert rc == 2
    assert "at least one mic channel" in capsys.readouterr().err


def test_mark_unknown_shot_errors(env, capsys, capture_reader):
    db, _ = env
    rc = workflow_cli.main(["mark", "999", "--ammo", "M855", "--se", "AI 1", "--db", db])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_mark_override_keys(env, capsys, capture_reader):
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    capsys.readouterr()
    rc = workflow_cli.main(
        ["mark", "1", "--ammo", "M855", "--se", "AI 1", "--sku", "SUP-9", "--db", db]
    )
    assert rc == 0
    assert "SKU SUP-9" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# list / report edge cases
# --------------------------------------------------------------------------- #


def test_list_groups_requires_batch(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["list", "groups", "--db", db])
    assert rc == 2
    assert "requires --batch" in capsys.readouterr().err


def test_list_groups_unknown_batch_errors(env, capsys):
    db, _ = env
    rc = workflow_cli.main(["list", "groups", "--batch", "42", "--db", db])
    assert rc == 2
    assert "No batch with id 42" in capsys.readouterr().err


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


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_show_reports_unset_then_set(env, capsys):
    db, inbox = env
    workflow_cli.main(["config", "show"])
    assert "(unset)" in capsys.readouterr().out

    workflow_cli.main(["config", "set-input-folder", str(inbox)])
    capsys.readouterr()
    workflow_cli.main(["config", "show"])
    out = capsys.readouterr().out
    assert str(inbox.resolve()) in out


def test_report_group_target(env, capsys, capture_reader):
    """`report --group` targets a single group directly."""
    db, inbox = env
    _touch(inbox, "SUP-1_AR15_001.dxd")
    workflow_cli.main(["ingest", str(inbox), "--db", db, "--no-validate"])
    workflow_cli.main(["mark", "1", "--ammo", "M855", "--se", "AI 1", "--db", db])
    capsys.readouterr()

    # Discover the group id via the repo, then report it.
    with WorkflowRepository(db) as repo:
        group_id = repo.groups_for_batch(1)[0].id
    assert workflow_cli.main(["report", "--group", str(group_id), "--db", db]) == 0
    assert "AR15 / M855" in capsys.readouterr().out
