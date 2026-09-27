"""Conditional-quantile curve over a metric's sample cloud — a visual aid only.

Feeds no reported number. Lives in the DSP layer (not the UI) so an export tool
can reproduce the same curve without PySide6.

Linear quantile regression (Koenker & Bassett 1978) on a cubic B-spline basis::

    rho(r) = r * (tau - 1{r < 0})     beta = argmin  sum_i rho(y_i - x_i' beta)

It traces the cloud's ``tau`` percentile at each instant — not a peak-hold or a
Hilbert envelope — which tolerates the cloud's very different spread across the
onset and down the decay.

Knots are uniform in a warped time centred on the onset,
``u(t) = sign(t - t0) * log1p(|t - t0| / eps)``: dense around the step, sparse
down the tail. Two guards, both recorded on the result:

* Each knot span must hold enough samples to estimate the quantile
  (:func:`_min_samples_per_span`). At :data:`DEFAULT_SAMPLES_PER_BASIS` this floor, not
  the warp, sets the density over most of the axis.
* Fitted values outside ``[min(y), max(y)]`` are basis ringing; they are clamped
  and ``clipped`` is set.

Solver: MM/IRLS from an OLS start, weights ``|tau - 1{r<0}| / max(|r|, delta)``,
banded normal equations. The fixed ``delta`` makes it limit-cycle at ~1e-2 dBA,
so ``converged`` is normally False; ``final_step`` reports the settled step and
the lowest-loss iterate is returned.

Not handled: separately fitted quantiles may cross. For a comparable family,
fit all at the ``n_basis`` the highest quantile resolves to (Bondell, Reich &
Wang 2010 for a guaranteed ordering).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solveh_banded

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from .graphing import MetricTrace

__all__ = [
    "DEFAULT_QUANTILE",
    "DEFAULT_SAMPLES_PER_BASIS",
    "DEFAULT_WARP_EPS_MS",
    "QUANTILE_METHOD",
    "QUANTILE_METRICS",
    "QuantileCurve",
    "fit_quantile_curve",
    "fit_trace_quantile",
]

#: Metrics drawn as a cloud with a spread to take a quantile of; the rest are
#: single-valued lines a quantile would only redraw.
QUANTILE_METRICS = frozenset({"liaeq_100ms_db"})
#: Tracks the top edge of an LIAeq dBA cloud without chasing outliers as 0.99 does.
DEFAULT_QUANTILE = 0.95
#: Coefficient budget, as samples per coefficient, so it scales with the capture
#: (an upper bound; the span floor caps it). Where tracking of a rolling local P95
#: flattens at both 100 and 210 ms frames: ~605 coefficients at 42 000 samples.
#: A fixed count coarsens the tail of a longer capture (1400 fitted 3.7 dBA off
#: at 210 ms against 2.1 here).
DEFAULT_SAMPLES_PER_BASIS = 14
#: Warp scale: near-linear within ``+/-eps`` of the onset, logarithmic beyond.
DEFAULT_WARP_EPS_MS = 0.05
_SPLINE_DEGREE = 3
#: Recorded on every result so an exported curve names its estimator.
QUANTILE_METHOD = "bspline-logwarp-irls"
#: IRLS weight denominator floor (check loss has no derivative at zero).
_WEIGHT_FLOOR = 1e-6
_MAX_ITER = 60
_TOL = 1e-8
#: Span floor is this over ``1 - tau``; 2 tracked the local quantile best.
_SPAN_SAMPLES_PER_TAIL = 2
#: Span floor for central quantiles, where the tail term shrinks toward nothing.
_SPAN_SAMPLES_MIN = 20
#: Drawn polyline: this many points in each of a warped and a real-time grid,
#: raised to ``_EVAL_POINTS_PER_SPAN`` per knot span for dense fits.
_EVAL_POINTS = 1500
_EVAL_POINTS_PER_SPAN = 10
#: Relative diagonal ridge; keeps a barely occupied basis column from going singular.
_RIDGE = 1e-9


def _warp(t_ms: np.ndarray, anchor_ms: float, eps_ms: float) -> np.ndarray:
    """Time to the knot coordinate; odd and monotone about the onset."""
    d = np.asarray(t_ms, dtype=float) - anchor_ms
    return np.sign(d) * np.log1p(np.abs(d) / eps_ms)


def _unwarp(u, anchor_ms: float, eps_ms: float) -> np.ndarray:
    """Inverse of :func:`_warp`."""
    u = np.asarray(u, dtype=float)
    return anchor_ms + np.sign(u) * eps_ms * np.expm1(np.abs(u))


def _min_samples_per_span(quantile: float) -> int:
    """Samples a knot span needs to estimate ``quantile``.

    Below ``1/(1 - tau)`` no sample can land above the curve; measured at
    tau = 0.95, coverage drops to 0.82 at the rank floor and recovers to 0.94 at
    ``2/(1 - tau)``. So: 20 for a median, 40 for Q95, 200 for Q99.
    """
    return max(
        _SPLINE_DEGREE + 1,
        _SPAN_SAMPLES_MIN,
        int(np.ceil(_SPAN_SAMPLES_PER_TAIL / max(1.0 - quantile, 1e-6))),
    )


def _min_knot_gap_ms(t_ms: np.ndarray, quantile: float = DEFAULT_QUANTILE) -> float:
    """Minimum breakpoint spacing in ms, off the median sample step."""
    if t_ms.size < 2:
        return 0.0
    step = float(np.median(np.diff(t_ms)))
    return max(step, 0.0) * _min_samples_per_span(quantile)


def _breakpoints(
    t_ms: np.ndarray,
    anchor_ms: float,
    eps_ms: float,
    n_basis: int,
    quantile: float = DEFAULT_QUANTILE,
) -> np.ndarray:
    """Breakpoints in warped coordinates, walked forward from the left edge.

    Each span is the wider of the uniform-in-``u`` step and the real-time floor
    (:func:`_min_knot_gap_ms`), so whichever limit binds locally applies.
    """
    u = _warp(t_ms, anchor_ms, eps_ms)
    u0, u1 = float(u[0]), float(u[-1])
    # n coefficients of degree k -> n - k spans.
    du = (u1 - u0) / max(1, n_basis - _SPLINE_DEGREE)
    gap = _min_knot_gap_ms(t_ms, quantile)
    kept = [u0]
    while True:
        previous = kept[-1]
        floor_u = _warp(
            np.array([_unwarp(previous, anchor_ms, eps_ms) + gap]), anchor_ms, eps_ms
        )[0]
        nxt = max(previous + du, float(floor_u))
        if nxt >= u1 or len(kept) >= n_basis:
            break
        kept.append(nxt)
    # The last span absorbs a remainder under the floor rather than leaving a sliver.
    if (
        len(kept) > 1
        and _unwarp(u1, anchor_ms, eps_ms) - _unwarp(kept[-1], anchor_ms, eps_ms) < gap
    ):
        kept.pop()
    kept.append(u1)
    return np.asarray(kept, dtype=float)


def _knot_vector(breakpoints: np.ndarray) -> np.ndarray:
    """Clamped knot vector: ends repeated ``degree`` extra times."""
    return np.concatenate(
        (
            np.repeat(breakpoints[0], _SPLINE_DEGREE),
            breakpoints,
            np.repeat(breakpoints[-1], _SPLINE_DEGREE),
        )
    )


def _check_loss(residual: np.ndarray, quantile: float) -> float:
    """Total check loss — the objective iterates are ranked by."""
    return float(np.sum(residual * (quantile - (residual < 0))))


def _weighted_fit(design, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least squares via banded Cholesky on ``X'WX``.

    The B-spline basis makes ``X'WX`` banded (``degree`` off-diagonals), so this
    is flat in sample count; dense solve measured 50-150 ms per call past ~100
    coefficients. Dense ``lstsq`` is only the fallback for a non-PD matrix.
    """
    scaled = design.multiply(weights[:, None]).tocsr()  # W X
    normal = (design.T @ scaled).tocsr()
    rhs = np.asarray(scaled.T @ y, dtype=float).ravel()
    size = normal.shape[0]
    # solveh_banded upper form: ab[degree + i - j, j] == A[i, j].
    ab = np.zeros((_SPLINE_DEGREE + 1, size))
    for offset in range(_SPLINE_DEGREE + 1):
        ab[_SPLINE_DEGREE - offset, offset:] = normal.diagonal(offset)
    scale = float(ab[_SPLINE_DEGREE].sum()) / max(size, 1)
    ridge = _RIDGE * max(scale, 1.0)
    ab[_SPLINE_DEGREE] += ridge
    try:
        return solveh_banded(ab, rhs)
    except (np.linalg.LinAlgError, ValueError):
        dense = np.asarray(normal.todense(), dtype=float)
        dense[np.diag_indices_from(dense)] += ridge
        return np.linalg.lstsq(dense, rhs, rcond=None)[0]


@dataclass(frozen=True)
class QuantileCurve:
    """A fitted quantile curve: the drawn polyline plus the model to resample it.

    ``t_ms`` is non-uniform (union of a warped and a real-time grid). Frozen so a
    cached fit can be shared between readers; derive a moved copy with
    :func:`dataclasses.replace`.
    """

    t_ms: np.ndarray
    values: np.ndarray
    quantile: float
    coefficients: np.ndarray
    #: Clamped knot vector in warped coordinates; see :attr:`knots_ms`.
    knots_u: np.ndarray
    anchor_ms: float
    warp_eps_ms: float
    degree: int = _SPLINE_DEGREE
    method: str = QUANTILE_METHOD
    #: Coefficients actually fitted after the span floor.
    n_basis: int = 0
    iterations: int = 0
    #: Normally False (IRLS limit-cycles); read :attr:`final_step` instead.
    converged: bool = False
    #: Last ``max|d beta|``; ~1e-2 dBA is a settled fit.
    final_step: float = float("nan")
    check_loss: float = float("nan")
    #: True if basis ringing was clamped to :attr:`data_range`.
    clipped: bool = False
    y_label: str = ""
    source_metric: str = ""
    label: str = ""
    data_range: tuple[float, float] = field(default=(float("nan"), float("nan")))

    @property
    def knots_ms(self) -> np.ndarray:
        return _unwarp(self.knots_u, self.anchor_ms, self.warp_eps_ms)

    def evaluate(self, t_ms) -> np.ndarray:
        """Resample the fitted curve; NaN outside the fitted span."""
        u = _warp(np.asarray(t_ms, dtype=float), self.anchor_ms, self.warp_eps_ms)
        spline = BSpline(self.knots_u, self.coefficients, self.degree, extrapolate=False)
        out = np.asarray(spline(u), dtype=float)
        lo, hi = self.data_range
        if np.isfinite(lo) and np.isfinite(hi):
            inside = np.isfinite(out)
            out[inside] = np.clip(out[inside], lo, hi)
        return out

    def as_dict(self) -> dict:
        """JSON-ready record: polyline, model, and fit diagnostics."""
        return {
            "method": self.method,
            "quantile": float(self.quantile),
            "source_metric": self.source_metric,
            "label": self.label,
            "y_label": self.y_label,
            "degree": int(self.degree),
            "n_basis": int(self.n_basis),
            "anchor_ms": float(self.anchor_ms),
            "warp_eps_ms": float(self.warp_eps_ms),
            "knots_u": [float(v) for v in self.knots_u],
            "coefficients": [float(v) for v in self.coefficients],
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "final_step": float(self.final_step),
            "check_loss": float(self.check_loss),
            "clipped": bool(self.clipped),
            "data_range": [float(self.data_range[0]), float(self.data_range[1])],
            "t_ms": [float(v) for v in self.t_ms],
            "values": [float(v) for v in self.values],
        }


def fit_quantile_curve(
    t_ms,
    values,
    *,
    quantile: float = DEFAULT_QUANTILE,
    anchor_ms: float | None = None,
    n_basis: int | None = None,
    warp_eps_ms: float = DEFAULT_WARP_EPS_MS,
    max_iter: int = _MAX_ITER,
    tol: float = _TOL,
    y_label: str = "",
    source_metric: str = "",
    label: str = "",
) -> QuantileCurve | None:
    """Fit ``Q_tau(values | t_ms)``; None if there is nothing to fit.

    ``anchor_ms`` should be the event onset; if None, the time of the largest
    sample is used. Non-finite samples are dropped. ``n_basis`` defaults to one
    coefficient per :data:`DEFAULT_SAMPLES_PER_BASIS` finite samples.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
    t = np.asarray(t_ms, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("t_ms and values must be matching 1-D arrays")

    finite = np.isfinite(t) & np.isfinite(y)
    if int(finite.sum()) < _SPLINE_DEGREE + 2:
        return None
    t, y = t[finite], y[finite]
    if t[-1] <= t[0]:
        return None
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi <= lo:
        return None

    if anchor_ms is None:
        anchor_ms = float(t[int(np.argmax(y))])
    anchor_ms = float(anchor_ms)
    if n_basis is None:
        n_basis = int(np.ceil(t.size / DEFAULT_SAMPLES_PER_BASIS))

    breakpoints = _breakpoints(t, anchor_ms, warp_eps_ms, n_basis, quantile)
    knots = _knot_vector(breakpoints)
    n_coef = knots.size - _SPLINE_DEGREE - 1
    if n_coef < _SPLINE_DEGREE + 1 or n_coef > t.size:
        return None

    u = _warp(t, anchor_ms, warp_eps_ms)
    design = BSpline.design_matrix(u, knots, _SPLINE_DEGREE, extrapolate=False).tocsr()

    beta = _weighted_fit(design, y, np.ones(t.size, dtype=float))  # OLS start
    # Keep the best iterate, not the last: the iteration limit-cycles.
    residual = y - np.asarray(design @ beta).ravel()
    best_beta, best_loss = beta, _check_loss(residual, quantile)
    iterations = 0
    converged = False
    step = float("nan")
    for iterations in range(1, max_iter + 1):
        weights = np.abs(quantile - (residual < 0)) / np.maximum(
            np.abs(residual), _WEIGHT_FLOOR
        )
        updated = _weighted_fit(design, y, weights)
        step = float(np.max(np.abs(updated - beta)))
        beta = updated
        residual = y - np.asarray(design @ beta).ravel()
        loss = _check_loss(residual, quantile)
        if loss < best_loss:
            best_beta, best_loss = beta, loss
        if step < tol:
            converged = True
            break
    beta = best_beta

    # Sample on warped + real-time grids so neither the onset nor a zoomed tail facets.
    points = max(_EVAL_POINTS, _EVAL_POINTS_PER_SPAN * n_coef)
    u_grid = np.linspace(u[0], u[-1], points)
    t_grid = _warp(np.linspace(t[0], t[-1], points), anchor_ms, warp_eps_ms)
    u_eval = np.unique(np.concatenate((u_grid, t_grid)))
    np.clip(u_eval, u[0], u[-1], out=u_eval)
    fitted = np.asarray(
        BSpline(knots, beta, _SPLINE_DEGREE, extrapolate=False)(u_eval), dtype=float
    )
    clipped = bool(np.any((fitted < lo) | (fitted > hi)))
    fitted = np.clip(fitted, lo, hi)

    return QuantileCurve(
        t_ms=_unwarp(u_eval, anchor_ms, warp_eps_ms),
        values=fitted,
        quantile=float(quantile),
        coefficients=beta,
        knots_u=knots,
        anchor_ms=anchor_ms,
        warp_eps_ms=float(warp_eps_ms),
        n_basis=int(n_coef),
        iterations=int(iterations),
        converged=converged,
        final_step=step,
        check_loss=best_loss,
        clipped=clipped,
        y_label=y_label,
        source_metric=source_metric,
        label=label,
        data_range=(lo, hi),
    )


def fit_trace_quantile(
    trace: MetricTrace,
    metric_key: str,
    *,
    quantile: float = DEFAULT_QUANTILE,
    label: str = "",
) -> QuantileCurve | None:
    """Fit ``trace`` the way the app does; None if ``metric_key`` takes no curve.

    The one place that decides which metrics are fitted (:data:`QUANTILE_METRICS`)
    and where a fit is anchored — the trace's calculation-window start, which is
    the detected onset — so any reader of a trace draws the same curve.
    """
    if metric_key not in QUANTILE_METRICS:
        return None
    anchor = (
        float(trace.t_ms[trace.window_start_index])
        if trace.window_start_index is not None
        else None
    )
    return fit_quantile_curve(
        trace.t_ms,
        trace.values,
        quantile=quantile,
        anchor_ms=anchor,
        y_label=trace.y_label,
        source_metric=metric_key,
        label=label,
    )
