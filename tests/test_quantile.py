"""Tests for the conditional-quantile curve fitter.

Central check is *coverage* (share of samples below the curve), per decade of
the decay as well as overall.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from sound_metric_app.config import EXPECTED_FS, EXPECTED_SAMPLES
from sound_metric_app.dsp import build_metric_trace
from sound_metric_app.dsp.quantile import (
    DEFAULT_QUANTILE,
    DEFAULT_SAMPLES_PER_BASIS,
    QUANTILE_METHOD,
    QUANTILE_METRICS,
    _min_knot_gap_ms,
    _unwarp,
    _warp,
    fit_quantile_curve,
    fit_trace_quantile,
)
from sound_metric_app.models import Frame

# A real 210 ms capture: density tuned on a shorter frame fits a longer one coarser.
FS = EXPECTED_FS
N = EXPECTED_SAMPLES
#: Pre-onset floor, the front, then the decades of the decay.
DECADES = ((0.0, 4.9), (6.0, 15.0), (15.0, 40.0), (40.0, 100.0), (100.0, 200.0))


def _shot_frame(onset_ms: float = 5.0, peak_pa: float = 300.0, seed: int = 7) -> Frame:
    """A blast with a noise floor, a fast front, and a two-tone ringing decay.

    The ringing is needed: without it a sparse fit tracks as well as a dense one.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS * 1000.0
    p = rng.normal(0.0, 0.002, N)
    start = int(onset_ms * FS / 1000.0)
    d = t[start:] - onset_ms
    p[start:] += peak_pa * np.exp(-d / 0.35) * (1 - d / 0.9)
    p[start:] += 0.25 * peak_pa * np.exp(-d / 12.0) * np.sin(2 * np.pi * d / 1.4)
    p[start:] += 0.08 * peak_pa * np.exp(-d / 30.0) * np.sin(2 * np.pi * d / 6.0)
    return Frame(samples=p, sample_rate=FS, channel="AI 1", source_file="x.dxd")


def _liaeq_trace(**kwargs):
    return build_metric_trace(_shot_frame(**kwargs), "liaeq_100ms_db")


def _fit(trace, **kwargs):
    anchor = float(trace.t_ms[trace.window_start_index])
    return fit_quantile_curve(trace.t_ms, trace.values, anchor_ms=anchor, **kwargs)


def _coverage(curve, t_ms, values, lo=None, hi=None) -> float:
    """Share of samples in ``[lo, hi)`` that fall below the fitted curve."""
    fitted = curve.evaluate(t_ms)
    mask = np.isfinite(fitted) & np.isfinite(values)
    if lo is not None:
        mask &= t_ms >= lo
    if hi is not None:
        mask &= t_ms < hi
    return float((values[mask] < fitted[mask]).mean())


def test_the_fit_covers_tau_of_the_cloud_in_every_decade_of_the_decay():
    trace = _liaeq_trace()
    curve = _fit(trace)
    assert curve is not None
    assert _coverage(curve, trace.t_ms, trace.values) == pytest.approx(
        DEFAULT_QUANTILE, abs=0.02
    )
    for lo, hi in DECADES:
        assert _coverage(curve, trace.t_ms, trace.values, lo, hi) == pytest.approx(
            DEFAULT_QUANTILE, abs=0.03
        ), f"coverage drifted over {lo}-{hi} ms"


def test_a_lower_quantile_estimates_its_own_level_and_sits_below_on_average():
    # No pointwise ordering check: independent fits may cross (see module doc).
    trace = _liaeq_trace()
    low = _fit(trace, quantile=0.5)
    high = _fit(trace, quantile=0.95)
    assert _coverage(low, trace.t_ms, trace.values) == pytest.approx(0.5, abs=0.03)
    assert _coverage(high, trace.t_ms, trace.values) == pytest.approx(0.95, abs=0.03)
    a, b = low.evaluate(trace.t_ms), high.evaluate(trace.t_ms)
    ok = np.isfinite(a) & np.isfinite(b)
    assert float(np.mean(b[ok] - a[ok])) > 0.5


def test_the_curve_never_claims_a_level_the_capture_did_not_reach():
    trace = _liaeq_trace()
    curve = _fit(trace)
    lo, hi = float(np.min(trace.values)), float(np.max(trace.values))
    assert curve.values.min() >= lo
    assert curve.values.max() <= hi
    assert curve.data_range == (lo, hi)
    # evaluate() clamps too.
    dense = curve.evaluate(np.linspace(trace.t_ms[0], trace.t_ms[-1], 5000))
    assert np.nanmax(dense) <= hi + 1e-9


def test_no_knot_span_is_finer_than_the_quantile_can_be_estimated_at():
    trace = _liaeq_trace(onset_ms=5.0)
    curve = _fit(trace)
    knots = np.unique(curve.knots_ms)
    spans = np.diff(knots)
    floor = _min_knot_gap_ms(trace.t_ms, DEFAULT_QUANTILE)
    assert float(np.min(spans)) >= floor - 1e-9
    # At the floor near the onset, wider down the far tail.
    near = spans[np.abs(knots[:-1] - 5.0) < 1.0]
    far = spans[knots[:-1] > 50.0]
    assert near.size and far.size
    assert float(np.median(near)) == pytest.approx(floor, rel=1e-6)
    assert float(np.median(far)) > float(np.median(near))


def test_a_far_tail_quantile_is_fitted_more_coarsely_than_a_central_one():
    # The span floor scales as 1/(1 - tau).
    trace = _liaeq_trace()
    coarse = _fit(trace, quantile=0.99)
    fine = _fit(trace, quantile=0.95)
    assert coarse.n_basis < fine.n_basis
    assert _min_knot_gap_ms(trace.t_ms, 0.99) > _min_knot_gap_ms(trace.t_ms, 0.95)


def _rolling_quantile(t_ms, values, window_ms, quantile=DEFAULT_QUANTILE):
    """Model-free empirical quantile in sliding windows: (centres, levels)."""
    step = float(np.median(np.diff(t_ms)))
    width = max(3, int(window_ms / step))
    centres, levels = [], []
    for start in range(0, values.size - width, max(1, width // 2)):
        centres.append(float(t_ms[start + width // 2]))
        levels.append(float(np.quantile(values[start : start + width], quantile)))
    return np.asarray(centres), np.asarray(levels)


def test_the_fit_tracks_the_local_quantile_closely_enough_to_show_ringing():
    # Coverage can't see a curve gliding through local peaks. Regression bound:
    # ~2.1 dBA now, 3.7 at the old fixed 1400-coefficient budget, ~6 sparser.
    trace = _liaeq_trace()
    curve = _fit(trace)
    centres, reference = _rolling_quantile(trace.t_ms, trace.values, 0.25)
    fitted = curve.evaluate(centres)
    ok = np.isfinite(fitted)
    rmse = float(np.sqrt(np.mean((fitted[ok] - reference[ok]) ** 2)))
    assert rmse < 3.0, f"the fit has been smoothed out: {rmse:.2f} dBA from local P95"
    sparse = _fit(trace, n_basis=40)
    sparse_fitted = sparse.evaluate(centres)
    ok_sparse = np.isfinite(sparse_fitted)
    sparse_rmse = float(
        np.sqrt(np.mean((sparse_fitted[ok_sparse] - reference[ok_sparse]) ** 2))
    )
    assert sparse_rmse > rmse * 1.5


def test_the_default_budget_scales_with_the_capture_not_a_fixed_count():
    # A fixed count (1400) spread a 210 ms frame's tail 2.2x coarser than a
    # 105 ms one's; per-sample budgeting holds it to ~1.1x.
    trace = _liaeq_trace()
    half = trace.t_ms.size // 2
    whole = _fit(trace)
    short = fit_quantile_curve(
        trace.t_ms[:half],
        trace.values[:half],
        anchor_ms=float(trace.t_ms[trace.window_start_index]),
    )
    assert whole.n_basis <= int(np.ceil(trace.t_ms.size / DEFAULT_SAMPLES_PER_BASIS))
    assert whole.n_basis > short.n_basis * 1.5
    tail = np.median(np.diff(np.unique(whole.knots_ms))[-5:])
    short_tail = np.median(np.diff(np.unique(short.knots_ms))[-5:])
    assert float(tail) < float(short_tail) * 1.5


def test_density_never_costs_the_fit_its_coverage():
    # Coverage drifted to 0.82 before the span floor scaled with the quantile.
    # None is the default budget; 4000 saturates the span floor.
    trace = _liaeq_trace()
    for n_basis in (40, 200, None, 4000):
        curve = _fit(trace, n_basis=n_basis)
        assert curve is not None
        assert curve.n_basis <= (n_basis or trace.t_ms.size)
        for lo, hi in DECADES:
            assert _coverage(curve, trace.t_ms, trace.values, lo, hi) == pytest.approx(
                DEFAULT_QUANTILE, abs=0.025
            ), f"n_basis={n_basis} drifted off tau over {lo}-{hi} ms"


def test_the_returned_fit_is_the_best_iterate_not_merely_the_last():
    trace = _liaeq_trace()
    curve = _fit(trace)
    assert curve.iterations > 0
    assert np.isfinite(curve.final_step)
    assert np.isfinite(curve.check_loss)
    assert curve.final_step < 0.5
    brief = _fit(trace, max_iter=5)
    assert curve.check_loss <= brief.check_loss


def test_the_warp_is_an_exact_invertible_change_of_variable():
    t = np.linspace(-20.0, 400.0, 1001)
    u = _warp(t, 5.0, 0.05)
    assert np.all(np.diff(u) > 0)
    assert _unwarp(u, 5.0, 0.05) == pytest.approx(t, abs=1e-9)


def test_the_fit_carries_everything_needed_to_reproduce_it_elsewhere():
    trace = _liaeq_trace()
    curve = _fit(trace, y_label=trace.y_label, source_metric="liaeq_100ms_db", label="#1 ML")
    record = json.loads(json.dumps(curve.as_dict()))
    assert record["method"] == QUANTILE_METHOD
    assert record["quantile"] == DEFAULT_QUANTILE
    assert record["source_metric"] == "liaeq_100ms_db"
    assert record["label"] == "#1 ML"
    assert record["y_label"] == trace.y_label
    assert len(record["coefficients"]) == record["n_basis"]
    assert len(record["knots_u"]) == record["n_basis"] + record["degree"] + 1
    assert len(record["t_ms"]) == len(record["values"])
    assert curve.evaluate(curve.t_ms) == pytest.approx(curve.values, abs=1e-9)
    assert np.all(np.isnan(curve.evaluate([trace.t_ms[0] - 1.0, trace.t_ms[-1] + 1.0])))


def test_the_fit_is_sampled_densely_at_the_onset_and_stays_ordered():
    trace = _liaeq_trace(onset_ms=5.0)
    curve = _fit(trace)
    assert np.all(np.diff(curve.t_ms) > 0)
    # Points per ms.
    near = int(np.sum(np.abs(curve.t_ms - 5.0) < 1.0)) / 2.0
    far = int(np.sum(curve.t_ms > 50.0)) / 50.0
    assert near > far * 10.0


def test_degenerate_input_yields_no_curve_rather_than_an_exception():
    t = np.linspace(0.0, 100.0, 500)
    assert fit_quantile_curve(t, np.zeros_like(t)) is None  # no spread
    assert fit_quantile_curve(t, np.full_like(t, np.nan)) is None  # nothing finite
    assert fit_quantile_curve(np.arange(3.0), np.arange(3.0)) is None  # too few
    assert fit_quantile_curve(np.zeros(50), np.linspace(0, 1, 50)) is None  # no span


def test_non_finite_samples_are_dropped_not_fitted():
    trace = _liaeq_trace()
    holed = trace.values.copy()
    holed[::7] = np.nan
    whole = _fit(trace)
    gapped = fit_quantile_curve(
        trace.t_ms, holed, anchor_ms=float(trace.t_ms[trace.window_start_index])
    )
    assert gapped is not None
    a, b = whole.evaluate(trace.t_ms), gapped.evaluate(trace.t_ms)
    ok = np.isfinite(a) & np.isfinite(b)
    assert float(np.median(np.abs(a[ok] - b[ok]))) < 1.0


def test_a_bad_quantile_is_refused_outright():
    t = np.linspace(0.0, 10.0, 100)
    y = np.linspace(0.0, 1.0, 100)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="quantile"):
            fit_quantile_curve(t, y, quantile=bad)
    with pytest.raises(ValueError, match="1-D arrays"):
        fit_quantile_curve(t, y[:50])


def test_fitting_does_not_touch_the_trace_it_reads():
    trace = _liaeq_trace()
    t_before, v_before = trace.t_ms.copy(), trace.values.copy()
    level_before = trace.level
    _fit(trace)
    assert trace.t_ms == pytest.approx(t_before)
    assert trace.values == pytest.approx(v_before, nan_ok=True)
    assert trace.level == level_before


def test_a_trace_is_fitted_anchored_at_its_window_start():
    trace = _liaeq_trace()
    curve = fit_trace_quantile(trace, "liaeq_100ms_db", label="#1 ML")
    expected = _fit(trace)
    assert curve.anchor_ms == pytest.approx(float(trace.t_ms[trace.window_start_index]))
    assert curve.values == pytest.approx(expected.values)
    assert curve.source_metric == "liaeq_100ms_db"
    assert curve.y_label == trace.y_label
    assert curve.label == "#1 ML"


def test_a_metric_drawn_as_a_line_takes_no_curve():
    trace = build_metric_trace(_shot_frame(), "peak_dba")
    assert "peak_dba" not in QUANTILE_METRICS
    assert fit_trace_quantile(trace, "peak_dba") is None


def test_a_fitted_curve_is_frozen():
    curve = _fit(_liaeq_trace())
    with pytest.raises(dataclasses.FrozenInstanceError):
        curve.anchor_ms = 0.0
    moved = dataclasses.replace(curve, anchor_ms=curve.anchor_ms + 1.0)
    assert moved.anchor_ms == pytest.approx(curve.anchor_ms + 1.0)


#: The only modules allowed to reach the quantile curve: the estimator, its
#: package re-export, and the Compare tab's graph. Batch reports, storage, the
#: CLI and exports must never draw on it.
_QUANTILE_ALLOWED = {
    "dsp/__init__.py",
    "dsp/quantile.py",
    "ui/graph/__init__.py",
    "ui/graph/metric_graph.py",
    "ui/graph/quantile_overlay.py",
    "ui/views/compare.py",
}


def _imports_quantile(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if "quantile" in (node.module or "").lower():
                return True
            if any("quantile" in alias.name.lower() for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any("quantile" in alias.name.lower() for alias in node.names):
                return True
    return False


def test_only_the_compare_graph_reaches_the_quantile_curve():
    package = Path(__file__).resolve().parents[1] / "src" / "sound_metric_app"
    offenders = sorted(
        rel
        for path in package.rglob("*.py")
        if (rel := path.relative_to(package).as_posix()) not in _QUANTILE_ALLOWED
        and _imports_quantile(ast.parse(path.read_text(encoding="utf-8")))
    )
    assert offenders == []
