# MATH

Mathematical definition of every metric produced by this application. Intended
for verification of correctness. Source of record: `src/sound_metric_app/dsp/`
and `src/sound_metric_app/config.py`.

## 1. Symbols and constants

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `p[n]` | Input pressure signal, one channel, Pascals | — | `Frame.samples` |
| `N` | Sample count per capture | 42 000 (nominal) | `EXPECTED_SAMPLES` |
| `fs` | Sample rate, Hz | 200 000 (nominal) | `EXPECTED_FS` |
| `T` | Capture duration, s | `N / fs` = 0.210 | `CAPTURE_MS` |
| `p_ref` | Reference pressure, Pa | 20 × 10⁻⁶ | `P_REF` |
| `p_A[n]` | A-weighted pressure signal | — | `apply_a_weighting` |
| `θ` | Onset threshold, Pa | per shot (default 2.0) | `Shot.trigger_pa` / `DEFAULT_TRIGGER_PA` |
| `θ₀` | Legacy onset fallback, Pa | 1.0 | `ONSET_THRESHOLD_PA` |
| `W_peak` | Peak/impulse search window, ms | 100 | `PEAK_WINDOW_MS` |
| `τ_L` | Leq rectangular integration time, s | 0.010 | `LEQ_TAU_S` |
| `W_Leq` | Peak-10 ms-Leq search window, ms | 25 | `LEQ_SEARCH_MS` |
| `W_LIAeq` | LIAeq energy window, ms | 100 | `LIAEQ_WINDOW_MS` |
| `τ_F` | Fast display time constant, s | 0.125 | `FAST_TIME_S` |
| `τ_S` | Slow display time constant, s | 1.0 | `SLOW_TIME_S` |
| `f1..f4` | A-weighting pole frequencies, Hz | 20.598997, 107.65265, 737.86223, 12194.217 | `weighting._F1.._F4` |

All decibel values are sound pressure levels (SPL) referenced to `p_ref`.

## 2. Assumptions

1. Input `p[n]` is calibrated absolute sound pressure in Pascals; no scaling or
   calibration is applied downstream of ingestion.
2. One capture file = one channel-frame. Metrics are stateless per frame; no
   filter or integrator state carries between frames.
3. Nominal acquisition is a **hardware trigger with a 10 ms pre-trigger lead and
   200 ms post-trigger capture** (`T = 210 ms`, `N = 42 000` at `fs = 200 kHz`).
   Actual `fs` and `N` from the file are used in all formulas; nominal values
   drive validation warnings only.
4. Reference pressure is `p_ref = 20 µPa` (air).
5. **Every metric is anchored to the shot onset** `n₀` (§3) and computed over a
   fixed window from there. `θ` is the trigger the shot was recorded with
   (`Shot.trigger_pa`, default 2 Pa; `θ₀ = 1 Pa` for a legacy shot that records
   none). A frame with no sample above `θ` is flagged and analysed from its
   start (the numbers are then suspect).
6. Metrics are computed independently per mic channel (SE, ML); channels are
   never combined at the DSP layer.
7. A-weighting follows IEC 61672 / ANSI S1.4, normalized to 0 dB at 1 kHz, and
   matches TBAC's `adsgn.m` (§8).
8. **One shot per analysis window.** The operator controls the range so no
   second blast, reflection, or comparable transient lands within `W = 100 ms`
   of onset.

## 3. Onset, windows, and base operators

**Onset.** `n₀ = min { n : p[n] > θ }`, the first sample whose *signed* raw
pressure exceeds `θ`, the shot's recorded trigger (`Shot.trigger_pa`, default
2 Pa; `θ₀ = 1 Pa` for a legacy shot). Every window below starts at `n₀`. If no
sample exceeds `θ`, `n₀ = 0` and a warning is emitted.

**Window operator.** For a signal `x` and width `w` ms:
```
W(x, w) = x[ n₀ : n₀ + round(w · fs / 1000) ]
```

**Signed peak** of a segment `x` — the positive-overpressure peak, *not* the
largest magnitude:
```
peak(x) = max_n x[n]           [Pa]
```

**Level of a linear magnitude** `v` (a pressure in Pa, or an impulse in Pa·ms):
```
L(v) = 20 · log10( v / p_ref )     [dB]        (−∞ if v ≤ 0)
```

**RMS** of a segment `x`:
```
rms(x) = sqrt( (1/M) · Σ_n x[n]² )     [Pa],  M = len(x)
```

## 4. Peak dB — `peak_pa`, `peak_db`

```
peak_pa = peak( W(p, W_peak) )                 [Pa]
peak_db = L(peak_pa)                            [dB]
```
Largest signed raw pressure in the 100 ms window after onset.

## 5. Peak dBA — `peak_a_pa`, `peak_dba`

```
peak_a_pa = peak( W(p_A, W_peak) )             [Pa]
peak_dba  = L(peak_a_pa)                        [dB]
```
Same operator on the A-weighted signal `p_A` (§8).

## 6. Peak Impulse — `impulse_pa_ms`, `peak_impulse_db`

The **positive-phase acoustic impulse** `∫p·dt` of the **unweighted** pressure,
in Pa·ms. Let `s = W(p, W_peak)` and `Δt = 1000 / fs` ms. Form the running
(cumulative-trapezoid) integral:
```
Q[0] = 0
Q[k] = Q[k−1] + (s[k−1] + s[k]) / 2 · Δt        [Pa·ms]
```
`Q` rises through the blast's positive-overpressure phase and falls once pressure
turns negative. The impulse is the peak of `Q` taken **before** its minimum (the
deepest point of the negative phase):
```
i_min         = argmin_k Q[k]
impulse_pa_ms = max( Q[0 .. i_min] )            [Pa·ms]   (global max if i_min = 0)
peak_impulse_db = L(impulse_pa_ms)              [dB·ms]
```
The `dB·ms` unit follows TBAC: `L(·)` of a Pa·ms magnitude, with time in
milliseconds.

## 7. LIAeq,100ms — `liaeq_pa`, `liaeq_100ms_db` (proprietary divergence)

A-weighted equivalent continuous level over the 100 ms free-field energy window
from onset:
```
liaeq_pa       = rms( W(p_A, W_LIAeq) )         [Pa]
liaeq_100ms_db = L(liaeq_pa) = 10 · log10( (1/M · Σ p_A²) / p_ref² )   [dB]
```

## 8. Peak Leq(10 ms) — `leq10ms_pa`, `leq10ms_db`

The maximum of a **10 ms rectangular running Leq** of the A-weighted signal,
searched within 25 ms of onset. First the rectangular running RMS, matching
Tougaard & Beedholm's `Leq_fast(..., 'rectangular')` with `L = floor(fs · τ_L)`
(= 2000 samples at 200 kHz) — a causal trailing moving mean-square, then root:
```
r[n] = sqrt( (1/L) · Σ_{k=n−L+1}^{n} p_A[k]² )      [Pa]   (causal; k<0 → 0)
```
The metric is the peak of `r` in the search window:
```
leq10ms_pa = max( W(r, W_Leq) )                 [Pa]
leq10ms_db = L(leq10ms_pa)                       [dB]
```

## 9. A-weighting filter — `a_weighting_sos` / `apply_a_weighting`

`p_A = A-weight(p)`. Analog IEC 61672 prototype in the Laplace domain: four zeros
at the origin and six real poles, gain set so the high-frequency asymptote matches
the standard.

**Zeros (s-plane):** `{0, 0, 0, 0}`
**Poles (s-plane, rad/s):** `−2π·f1 (×2), −2π·f2, −2π·f3, −2π·f4 (×2)`
**Analog gain:** `k = (2π·f4)²`

```
              k · s⁴
H(s) = ────────────────────────────────────────────────
        (s + 2π·f1)² (s + 2π·f2)(s + 2π·f3)(s + 2π·f4)²
```

**Discretization:** bilinear transform of `(zeros, poles, k)` at rate `fs`
(`scipy.signal.bilinear_zpk`), converted to second-order sections (`zpk2sos`).

**Normalization:** the first section's numerator is scaled so the discrete filter
magnitude at 1 kHz is exactly 1 (0 dB):
```
sos[0, 0:3] ← sos[0, 0:3] / |H_d(e^{j2π·1000/fs})|
```

**Application:** causal IIR filtering `p_A = sosfilt(sos, p)` (forward only).

**Verification points** (relative response, from `tests/test_metrics.py`):

| Frequency | Expected A-weighting | Tolerance |
|---|---|---|
| 100 Hz | −19.1 dB | ±0.5 dB |
| 1 kHz | 0.0 dB | ±0.1 dB |
| 10 kHz | −2.5 dB | ±0.7 dB |

## 10. Batch aggregation — `repository.batch_averages`

Per batch (one test session under a fixed Suppressor SKU + Test Platform + Ammo
combination), the **included** shots are partitioned by mic position crossed with
derived role, giving four output slots:

```
muzzle_left  (ML) · FRP        muzzle_left  (ML) · regular
shooters_ear (SE) · FRP        shooters_ear (SE) · regular
```

A shot's role is derived from its order within its cluster — order 0 is the FRP
(first round pop), everything after is regular — so a cluster has exactly one FRP
by construction. Only shots whose `included` flag is set contribute; the rest stay
in the data bank untouched.

Within each slot, each metric is averaged in the **linear domain** — the per-shot
linear magnitudes (Pa, or Pa·ms for the impulse) are meaned, then the mean is
converted once to its dB level:
```
metric_avg_linear = (1/n) · Σ_{included shots in slot} metric_shot_linear
metric_avg_db     = L( metric_avg_linear )
```
where the linear magnitude is one of `{peak_pa, peak_a_pa, impulse_pa_ms,
leq10ms_pa, liaeq_pa}` and `n` is the included-shot count for that slot. The four
slots are computed as separate `GROUP BY mic_position, role` partitions and never
combined: positions are not mixed, and neither are roles. Each metric's average
skips shots whose value is missing (an unpopulated row after a migration, or a NaN
metric stored as NULL); in normal fully-populated operation that count equals `n`,
so the two coincide.

## 11. Fast/Slow display envelope — `graphing._exp_rms_spl_db`

Display-only smoothing for the SPL-over-time report graphs; it does **not** feed
any stored metric. Squared pressure is passed through a one-pole exponential
average (IEC 61672 Fast `τ_F = 125 ms` / Slow `τ_S = 1 s`), then converted to dB:
```
a       = exp( −1 / (fs · τ) )              τ ∈ {τ_F, τ_S}
y[n]    = a · y[n−1] + (1 − a) · x_sig[n]²
L[n]    = 10 · log10( max(y[n], p_ref²) / p_ref² )        [dB]
```
The mean square is floored at `p_ref²` (0 dB) so silent stretches read as a clean
0 dB floor instead of −∞. The A-weighted trace passes `x_sig = p_A`; the
unweighted trace passes `x_sig = p`. A Pascal-domain variant (`_exp_rms_pa`)
returns `sqrt(max(y, 0))` in Pa without the dB conversion.
