"""Unit tests for the hierarchy models and the capture-filename parser."""

from __future__ import annotations

import pytest

from sound_metric_app.models import (
    Batch,
    Group,
    MetricResult,
    MicPosition,
    ParsedCaptureName,
    Shot,
    parse_capture_filename,
)


# --- parse_capture_filename ------------------------------------------------- #


@pytest.mark.parametrize(
    "name, expected",
    [
        ("SUP-1234_AR15_003.dxd", ("SUP-1234", "AR15", 3)),
        ("SUP-1234_AR15_003.d7d", ("SUP-1234", "AR15", 3)),
        ("SUP-1234_AR15_003.DXD", ("SUP-1234", "AR15", 3)),  # case-insensitive ext
        ("SUP-1234_AR15_003", ("SUP-1234", "AR15", 3)),  # no extension
        ("/input/SUP-9_M4_017.dxd", ("SUP-9", "M4", 17)),  # full path (POSIX sep parses on both OSes)
        ("SUP-1234_AR15_000.dxd", ("SUP-1234", "AR15", 0)),  # zero order
    ],
)
def test_parse_valid_names(name, expected):
    parsed = parse_capture_filename(name)
    assert parsed == ParsedCaptureName(*expected)
    # Unpacks as a plain 3-tuple too.
    sku, platform, order = parse_capture_filename(name)
    assert (sku, platform, order) == expected
    assert isinstance(order, int)


@pytest.mark.parametrize(
    "name",
    [
        "SUP-1234_AR15.dxd",  # too few fields
        "SUP-1234_AR15_003_extra.dxd",  # too many fields
        "SUP-1234_AR15_003.txt",  # wrong extension
        "SUP-1234_AR15_x03.dxd",  # non-numeric order
        "SUP-1234_AR15_².dxd",  # digit-category char int() rejects
        "SUP-1234__003.dxd",  # empty platform
        "_AR15_003.dxd",  # empty sku
        "SUP-1234_AR15_.dxd",  # empty order
        "no_underscores_here_at_all.dxd",  # 5 fields
    ],
)
def test_parse_rejects_malformed(name):
    with pytest.raises(ValueError):
        parse_capture_filename(name)


def test_parse_unicode_digit_gives_friendly_error():
    # str.isdigit() is True for '²' but int() rejects it; the friendly
    # "is not numeric" message must win over a raw int() ValueError.
    with pytest.raises(ValueError, match="is not numeric"):
        parse_capture_filename("SUP_AR15_²")


# --- enum + dataclasses ----------------------------------------------------- #


def test_mic_position_is_str_enum():
    assert MicPosition.SE.value == "SE"
    assert MicPosition("MR") is MicPosition.MR
    # str mixin: value round-trips to text storage directly.
    assert MicPosition.SE == "SE"


def test_batch_group_shot_defaults():
    batch = Batch(sku="SUP-1234")
    assert batch.closed is False and batch.id is None

    group = Group(test_platform="AR15", ammo="M855")
    assert group.batch_id is None

    shot = Shot(source_file="SUP-1234_AR15_003.dxd", suppressor_sku="SUP-1234")
    # Unmarked by default; group/marking fields empty until marked.
    assert shot.marked is False
    assert shot.ammo is None and shot.group_id is None
    assert shot.se_channel is None and shot.mr_channel is None


def test_metric_result_as_row_has_no_mic_position():
    # Mic position is not a DSP-result concern; storage receives it separately.
    r = MetricResult(
        peak_db=1.0,
        peak_dba=2.0,
        peak_impulse_db=3.0,
        laimax_db=3.5,
        liaeq_100ms_db=4.0,
        source_file="f.dxd",
        channel="AI 1",
        sample_rate=200_000.0,
        n_samples=20_000,
    )
    assert "mic_position" not in r.as_row()
