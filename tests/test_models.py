"""Unit tests for the hierarchy models and the capture-filename parser."""

from __future__ import annotations

import pytest

from sound_metric_app.models import (
    Batch,
    Cluster,
    Combination,
    MetricResult,
    MicPosition,
    ParsedCaptureName,
    Shot,
    ShotRole,
    parse_capture_filename,
    role_for_order,
)


# --- parse_capture_filename ------------------------------------------------- #


@pytest.mark.parametrize(
    "name, expected",
    [
        ("SUP-1234_AR15_02_003.dxd", ("SUP-1234", "AR15", 2, 3)),
        ("SUP-1234_AR15_02_003.d7d", ("SUP-1234", "AR15", 2, 3)),
        ("SUP-1234_AR15_02_003.DXD", ("SUP-1234", "AR15", 2, 3)),  # case-insensitive ext
        ("SUP-1234_AR15_02_003", ("SUP-1234", "AR15", 2, 3)),  # no extension
        # full path (POSIX sep parses on both OSes)
        ("/input/SUP-9_M4_1_017.dxd", ("SUP-9", "M4", 1, 17)),
        ("SUP-1234_AR15_10_001.dxd", ("SUP-1234", "AR15", 10, 1)),  # multi-digit cluster
        # Dewesoft counts its exports from zero, so the FRP arrives as 0000.
        ("SUP-1234_AR15_02_0000.dxd", ("SUP-1234", "AR15", 2, 0)),
        ("SUP-1234_AR15_02_0001.dxd", ("SUP-1234", "AR15", 2, 1)),
    ],
)
def test_parse_valid_names(name, expected):
    parsed = parse_capture_filename(name)
    assert parsed == ParsedCaptureName(*expected)
    # Unpacks as a plain 4-tuple too.
    sku, platform, cluster, order = parse_capture_filename(name)
    assert (sku, platform, cluster, order) == expected
    assert isinstance(cluster, int) and isinstance(order, int)


@pytest.mark.parametrize(
    "name",
    [
        "SUP-1234_AR15_003.dxd",  # too few fields (the old 3-field convention)
        "SUP-1234_AR15.dxd",  # far too few
        "SUP-1234_AR15_02_003_extra.dxd",  # too many fields
        "SUP-1234_AR15_02_003.txt",  # wrong extension
        "SUP-1234_AR15_02_x03.dxd",  # non-numeric order
        "SUP-1234_AR15_xx_003.dxd",  # non-numeric cluster
        "SUP-1234_AR15_02_².dxd",  # digit-category char int() rejects
        "SUP-1234__02_003.dxd",  # empty platform
        "_AR15_02_003.dxd",  # empty sku
        "SUP-1234_AR15_02_.dxd",  # empty order
    ],
)
def test_parse_rejects_malformed(name):
    with pytest.raises(ValueError):
        parse_capture_filename(name)


def test_parse_rejects_zero_cluster():
    # Cluster indices are ours, not Dewesoft's, and stay 1-based: a 0 would not
    # name a real string of fire. Shot orders are the opposite — see below.
    with pytest.raises(ValueError, match="must be 1 or greater"):
        parse_capture_filename("SUP-1_AR15_00_0003.dxd")


def test_parse_accepts_zero_shot_order_as_the_frp():
    # DewesoftX numbers its exports from zero, so the first round of a string
    # lands as _0000 and that is exactly the shot we call the FRP.
    assert parse_capture_filename("SUP-1_AR15_01_0000.dxd").shot_order == 0
    assert role_for_order(0) is ShotRole.FRP


def test_parse_unicode_digit_gives_friendly_error():
    # str.isdigit() is True for '²' but int() rejects it; the friendly
    # "is not numeric" message must win over a raw int() ValueError.
    with pytest.raises(ValueError, match="is not numeric"):
        parse_capture_filename("SUP_AR15_1_²")


# --- roles ------------------------------------------------------------------ #


def test_role_is_derived_from_shot_order():
    assert role_for_order(0) is ShotRole.FRP
    assert role_for_order(1) is ShotRole.REGULAR
    assert role_for_order(99) is ShotRole.REGULAR
    # No order means no derivable role — such a shot cannot enter an average.
    assert role_for_order(None) is None


def test_shot_role_property_tracks_shot_order():
    shot = Shot(source_file="f.dxd", shot_order=0)
    assert shot.role is ShotRole.FRP
    # Role is a property, not stored state: re-ordering the shot re-derives it,
    # which is what guarantees exactly one FRP per cluster.
    shot.shot_order = 3
    assert shot.role is ShotRole.REGULAR


# --- enum + dataclasses ----------------------------------------------------- #


def test_mic_position_is_str_enum():
    assert MicPosition.SE.value == "SE"
    assert MicPosition("ML") is MicPosition.ML
    # str mixin: value round-trips to text storage directly.
    assert MicPosition.SE == "SE"
    assert MicPosition.ML.label == "Muzzle Left"
    assert MicPosition.SE.label == "Shooter's Ear"


def test_combination_label_is_the_three_test_conditions():
    combo = Combination(sku="SUP-1234", platform="AR15", ammo="5.56 M855")
    assert combo.label == "SUP-1234 / AR15 / 5.56 M855"


def test_batch_title_falls_back_to_a_placeholder():
    # `title`, not `label`: a batch's `label` is the user's name for the session.
    assert Batch(combination_id=1, label="Morning", session_date="2026-07-22").title == (
        "Morning 2026-07-22"
    )
    assert Batch(combination_id=1, label="Morning").title == "Morning"
    assert Batch(combination_id=1, session_date="2026-07-22").title == "2026-07-22"
    assert Batch(combination_id=1).title == "(unnamed session)"


def test_batch_weather_summary_omits_unrecorded_fields():
    batch = Batch(combination_id=1, wind_speed=3.5, temp=71, relative_humidity=40)
    assert batch.weather_summary == "wind 3.5 mph, 71 °F, RH 40%"
    assert Batch(combination_id=1, temp=71).weather_summary == "71 °F"
    # Empty, not a placeholder — each surface supplies its own.
    assert Batch(combination_id=1).weather_summary == ""


def test_hierarchy_defaults():
    # A batch is a session under a combination; every session field is optional
    # because the batch is created the moment its first shot is marked.
    batch = Batch(combination_id=1)
    assert batch.closed is False and batch.id is None
    assert batch.session_date is None and batch.notes is None

    cluster = Cluster(batch_id=1, cluster_index=2)
    assert cluster.label == "Cluster 2"

    shot = Shot(source_file="SUP-1234_AR15_02_003.dxd", suppressor_sku="SUP-1234")
    # Unmarked by default; placement/marking fields empty until marked.
    assert shot.marked is False
    assert shot.ammo is None and shot.cluster_id is None
    assert shot.se_channel is None and shot.ml_channel is None
    # Idle by default: a shot must be explicitly brought forward to be averaged.
    assert shot.included is False and shot.exclusion_reason is None


def test_metric_result_as_row_has_no_mic_position():
    # Mic position is not a DSP-result concern; storage receives it separately.
    r = MetricResult(
        peak_pa=5.0, peak_db=1.0,
        peak_a_pa=6.0, peak_dba=2.0,
        impulse_pa_ms=7.0, peak_impulse_db=3.0,
        leq10ms_pa=8.0, leq10ms_db=9.0,
        liaeq_pa=10.0, liaeq_100ms_db=4.0,
        source_file="f.dxd",
        channel="AI 1",
        sample_rate=200_000.0,
        n_samples=42_000,
    )
    row = r.as_row()
    assert "mic_position" not in row
    # Every metric carries both its linear magnitude and its dB level.
    for key in ("peak_pa", "peak_a_pa", "impulse_pa_ms", "leq10ms_pa", "leq10ms_db", "liaeq_pa"):
        assert key in row
