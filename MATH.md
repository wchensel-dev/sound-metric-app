# MATH

Mathematical definition of every metric produced by this application. Intended
for verification of correctness. Source of record:
`src/sound_metric_app/dsp/` and `src/sound_metric_app/config.py`.

## 1. Symbols and constants

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `p[n]` | Input pressure signal, one channel, Pascals | — | `Frame.samples` |
| `N` | Sample count per frame | 20 000 (nominal) | `EXPECTED_SAMPLES` |
| `fs` | Sample rate, Hz | 200 000 (nominal) | `EXPECTED_FS` |
| `T` | Frame duration, s | `N / fs` = 0.100 | derived |
| `p_ref` | Reference pressure, Pa | 20 × 10⁻⁶ | `P_REF` |
| `p_A[n]` | A-weighted pressure signal | — | `apply_a_weighting` |
| `τ_r` | Impulse rise time constant, s | 0.035 | `IMPULSE_RISE_S` |
| `τ_f` | Impulse fall time constant, s | 1.5 | `IMPULSE_FALL_S` |
| `f1..f4` | A-weighting pole frequencies, Hz | 20.598997, 107.65265, 737.86223, 12194.217 | `weighting._F1.._F4` |

All decibel values are sound pressure levels (SPL) referenced to `p_ref`.

## 2. Assumptions

1. Input `p[n]` is calibrated absolute sound pressure in Pascals; no scaling or
   calibration is applied downstream of ingestion.
2. One capture file = one channel-frame of `N` samples at `fs`. Metrics are
   stateless per frame; no filter or integrator state carries between frames.
3. Nominal `fs = 200 kHz`, `N = 20 000` (`T = 100 ms`). Actual `fs` from the
   file is used in all formulas; nominal values drive validation warnings only.
4. Reference pressure is `p_ref = 20 µPa` (air).
5. Metrics are computed independently per mic channel (SE, MR); channels are
   never combined at the DSP layer.
6. A-weighting follows IEC 61672 / ANSI S1.4, normalized to 0 dB at 1 kHz.
7. **Validation status (against the DewesoftX sound-level plugin,
   `CLAPDS_556_0001.dxd`):** Peak dB (§4) and Peak dBA (§5) are exact. The
   A-weighting filter (§8) matches Dewesoft's `(A)` channel at correlation
   0.999989. The single-stage Impulse detector (§6) reproduces Dewesoft's `LAIp`
   channel, and its peak-hold **LAImax (§6.1) matches to 4 decimals** — so the
   detector and its 35 ms / 1500 ms constants are validated, even though it is
   nominally simpler than the two-stage IEC 61672 detector. The `peak_impulse_db`
   dB·ms **integral** (§6) remains a home-grown quantity with no meter
   counterpart. `LIAeq,100ms` (§7) has exact arithmetic but a different
   averaging-window reference than Dewesoft's session `LAeq` — see §7's note.

## 3. Base operators

**Peak level** of a signal `x`:

```
L_peak(x) = 20 · log10( max_n |x[n]| / p_ref )        [dB]
```
Returns −∞ if `max |x| = 0`.

**Equivalent (RMS) level** of a signal `x` over its full length `N`:

```
L_eq(x) = 10 · log10( (1/N · Σ_n x[n]²) / p_ref² )    [dB]
```
Equivalently `20 · log10(rms(x) / p_ref)`. Returns −∞ if the mean square is 0.

## 4. Peak dB — `peak_db` (stable)

```
Peak_dB = L_peak(p) = 20 · log10( max_n |p[n]| / p_ref )
```
Unweighted peak of the raw pressure signal.

## 5. Peak dBA — `peak_dba` (stable)

```
Peak_dBA = L_peak(p_A) = 20 · log10( max_n |p_A[n]| / p_ref )
```
Same peak operator applied to the A-weighted signal `p_A` (§8).

## 6. Impulse — `peak_impulse_db` (provisional)

Computed on the **A-weighted** signal `p_A`. Units: **dB·ms**.

First form the instantaneous Impulse ("I") time-weighted level, sample by
sample, via a one-pole exponential smoother of squared pressure with an
asymmetric (fast-attack / slow-release) time constant:

```
α_r = exp( −1 / (fs · τ_r) )        (rise / attack coefficient)
α_f = exp( −1 / (fs · τ_f) )        (fall / release coefficient)

x[n] = p_A[n]²

s[-1] = 0
for n = 0 .. N−1:
    α    = α_r   if x[n] > s[n−1]   else α_f
    s[n] = α · s[n−1] + (1 − α) · x[n]

L_I[n] = 10 · log10( s[n] / p_ref² )        [dB]
```

The metric is the **time integral** of `L_I` over the frame, evaluated by
forward-Euler (rectangular) numerical integration with step `Δt = 1000 / fs`
milliseconds:

```
Impulse = Σ_{n=0}^{N−1} L_I[n] · Δt          [dB·ms]
```

This is why the reported quantity carries units of **dB·ms** (a dB level
integrated over time) rather than the plain dB of a peak level: the `· Δt`
factor supplies the millisecond dimension. Samples where `s[n] = 0` (so
`L_I[n] = −∞`) are omitted from the sum, so a silent frame integrates to `0`.

Notes:
- `s[n]` is a smoothed mean-square estimate; attack uses `τ_r = 35 ms`, release
  uses `τ_f = 1500 ms`.
- The integration runs over the **whole 100 ms frame**, so the peak of `L_I`
  is always included in the total.
- This is a **single-stage** exponential detector, not the two-stage
  (RMS-detector followed by peak-hold) Impulse detector of IEC 61672 — the
  simplification noted in §2.7.

## 6.1 LAImax — `laimax_db` (validated)

The **peak-hold** of the Impulse time-weighted level over the frame, on the
same `L_I[n]` from §6:

```
LAImax = max_n L_I[n]          [dB(A)]
```

Unlike `peak_impulse_db` (the dB·ms time integral, a home-grown quantity), this
is the reading a sound-level meter's "I" detector reports and is directly
comparable to instrumentation. A silent frame (every `L_I[n] = −∞`) yields
`−∞`; a NaN in the input propagates so contaminated data surfaces.

**Validation:** against the DewesoftX sound-level plugin's `LAIp` channel for
`CLAPDS_556_0001.dxd`, `LAImax` matches the maximum of `LAIp` to **4 decimals**
(105.3346 dB(A) computed vs 105.3349 reported) when our detector is fed
Dewesoft's own A-weighted signal — confirming both the §6 detector and its
τ_r = 35 ms / τ_f = 1500 ms constants. See §2.7.

## 7. LIAeq,100ms — `liaeq_100ms_db` (validated math; see averaging-window note)

A-weighted equivalent continuous level over the whole frame:

```
LIAeq_100ms = L_eq(p_A) = 10 · log10( (1/N · Σ_n p_A[n]²) / p_ref² )
```
The averaging window is the entire frame (`T = 100 ms` nominal).

**Validation / averaging-window note:** the mean-square→dB arithmetic is exact —
fed Dewesoft's own A-weighted signal it reproduces Dewesoft's energy to the
rounding. It does **not** match Dewesoft's `LAeq` channel value directly because
the two use different reference times: our `T = 100 ms` frame vs Dewesoft's full
measurement session. On `CLAPDS_556_0001.dxd` the gap is exactly
`10 · log10(4.0 s / 0.1 s) = 16.0 dB` (ours 101.4, Dewesoft 85.4). The per-frame
reference is the intended quantity here — a per-shot metric must not depend on
how long the meter ran — so this is a definitional difference, not an error.

## 8. A-weighting filter — `a_weighting_sos` / `apply_a_weighting`

`p_A = A-weight(p)`. Analog IEC 61672 prototype in the Laplace domain: four
zeros at the origin and six real poles, with gain set so the high-frequency
asymptote matches the standard.

**Zeros (s-plane):** `{0, 0, 0, 0}`

**Poles (s-plane, rad/s):**
```
−2π·f1  (×2),  −2π·f2,  −2π·f3,  −2π·f4  (×2)
```

**Analog gain:**
```
k = (2π·f4)²
```

Transfer function form:

```
              k · s⁴
H(s) = ────────────────────────────────────────────────
        (s + 2π·f1)² (s + 2π·f2)(s + 2π·f3)(s + 2π·f4)²
```

**Discretization:** bilinear transform of `(zeros, poles, k)` at rate `fs`
(`scipy.signal.bilinear_zpk`), converted to second-order sections
(`zpk2sos`).

**Normalization:** the first section's numerator is scaled so the discrete
filter magnitude at 1 kHz is exactly 1 (0 dB):

```
sos[0, 0:3] ← sos[0, 0:3] / |H_d(e^{j2π·1000/fs})|
```

**Application:** causal IIR filtering `p_A = sosfilt(sos, p)` (forward only, not
zero-phase).

**Verification points** (relative response, from `tests/test_metrics.py`):

| Frequency | Expected A-weighting | Tolerance |
|---|---|---|
| 100 Hz | −19.1 dB | ±0.5 dB |
| 1 kHz | 0.0 dB | ±0.1 dB |
| 10 kHz | −2.5 dB | ±0.7 dB |

## 9. Group aggregation — `repository.group_averages`

Per group (fixed Suppressor SKU + Test Platform + Ammo) and per mic position
(SE, MR kept separate), each stored metric is averaged as a **plain arithmetic
mean of the decibel values** across the group's shots:

```
metric_avg = (1/n) · Σ_{shots in group, matching position} metric_shot
```
where `metric ∈ {peak_db, peak_dba, peak_impulse_db, liaeq_100ms_db}` and `n`
is the shot count for that position. SE and MR are aggregated in separate
`GROUP BY mic_position` partitions and never combined. The dB-level metrics
average in `[dB]`; `peak_impulse_db` averages in `[dB·ms]` (§6).

Note: averaging is performed in the **dB (log) domain**, not on linear pressure
or energy. This is the store's stated convention, not an energy-equivalent mean.
`peak_impulse_db` is a dB level already integrated over time (§6); its shots are
likewise averaged arithmetically.
