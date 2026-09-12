# MATHWORK

This document holds the contextual, rationale, and development-tracking prose
that was pulled out of `MATH.md` so that file can stay a dry, implementation-
agnostic snapshot of the metric mathematics. Each block below is keyed to the
`MATH.md` section (`§N`) it came from and reproduces that text verbatim. It
records intent, design rationale, history, edge-case discussion, and the
relationship to the TBAC reference — material for developers and future
modification, not part of the formal metric definitions.

## Preamble

The metric definitions follow Thunder Beast Arms Corp's (TBAC) `process_string.m`
reference, with the deliberate divergences noted in §12.

## §2. Assumptions

The trigger level was historically 1 Pa, was raised to 10 Pa by the techs and
is now 2 Pa to reject wind gusts; it is not stored in the file.

This assumes the pre-trigger baseline is quiet relative to `θ` so the
first threshold crossing is the shot, not noise — recording at the recorder's
own trigger keeps sub-trigger wind in the lead from capturing the onset.

All onset-anchored metrics share that one window (§3), so this
assumption is what makes "largest sample in the window" (§4) and "the shot's
peak" the same quantity, and what keeps `LIAeq,100ms` (§7) attributable to a
single shot. A contaminating event is not truncated away — an event inside
the window is inside every metric, by design, so a frame that violates this
is visible rather than silently partitioned.

## §3. Onset, windows, and base operators

This generalises TBAC's `find(Y>1.)`, which is the `θ = 1 Pa` case.

## §4. Peak dB

This is the
largest sample *in the window*, not "the shot's peak" by construction — the two
coincide because capture discipline keeps one shot per frame (§2.8).

## §6. Peak Impulse

so a later secondary rise cannot inflate it (TBAC's dynamic window)

Because every metric is onset-anchored, `Q` starts at the shot, so
its positive phase is contiguous from `k = 0` and the peak is captured.

The min-bounding rejects a later (e.g. reflected) rise **only when the rarefaction
drives `Q` below its start** (`i_min > 0`) — the usual free-field case. When `Q`
stays non-negative over the whole window (`i_min = 0`, e.g. a blast whose
rarefaction never pulls the running integral negative), the impulse is the global
max over the **entire `W_peak = 100 ms` window**, and any rise within those 100 ms
— a reflection, a second blast — could in principle inflate it. The capture
discipline of §2.8 (one shot, no comparable transient within `W` of onset) is the
*only* thing that bounds it. Nothing in the code detects or flags an `i_min = 0`
frame, so a violation of §2.8 shows up as a silently high impulse rather than a
warning — inspect the Report graph's `Q` trace (which marks the peak) when an
impulse reads implausibly high. TBAC clips to a short window instead for exactly
this reason in a reverberant space (§12). A NaN in the input propagates so
contaminated data surfaces.

## §7. LIAeq,100ms

This is our divergence from TBAC (§12): where they take a peak 10 ms-Leq (§8.1)
to reject reflections in a reverberant space, we integrate the full 100 ms of the
free-field decay. Both are reported so shots validate against TBAC and against
our model on the same capture.

## §8. Peak Leq(10 ms)

Unlike `Leq_fast`'s FFT (circular) convolution, `r` is strictly causal, so its
first `L` samples ramp up from zero state instead of wrapping the array tail; the
onset-anchored search window sits past that ramp, so the reported maximum matches.

## §9. A-weighting filter

**Parity with TBAC.** TBAC's `adsgn.m` (Couvreur, IEC 1672) is the *same* analog
prototype — identical `f1..f4`, four zeros at the origin, identical pole
structure. It differs only in how 1 kHz is normalized: TBAC bakes in the analytic
constant `A1000 = 1.9997 dB` (numerator × 10^(1.9997/20) ≈ × 1.2589), where we
measure the discrete 1 kHz response and divide. The two agree to sub-millidecibel
at these sample rates. (TBAC's `bilinear(..., 1/Fs)` passes the sampling period
`T` per Octave's convention, not a bug; scipy passes `fs`.)

## §10. Batch aggregation

This matches TBAC, which accumulates linear Pa (and Pa·ms) across shots, divides
by the shot count, and converts to dB at the end. It is **not** a mean of the dB
values (which, by Jensen, would read lower); the log is applied once, to the mean.

## §11. Fast/Slow display envelope

This is the
exact-normalization form of the FFT-based `Leq_fast` running-RMS routine
(Tougaard & Beedholm, 2018); the two normalizations differ by ~3 × 10⁻⁵ dB.

## §12. Divergences from TBAC

| Axis | TBAC `process_string.m` | This app | Kind |
|---|---|---|---|
| Sample rate | 262 144 Hz (2¹⁷ per 0.5 s) | 200 000 Hz clean | deliberate |
| Analysis anchor | peak/impulse from fixed `Time_Start`; Leq from onset | **all windows onset-anchored** | deliberate (robustness) |
| Peak | signed positive overpressure | signed positive overpressure | aligned |
| Impulse | `∫p·dt` positive phase, unweighted, Pa·ms + dB·ms | same | aligned |
| Peak 10 ms-Leq | max 10 ms rectangular running Leq within 25 ms of onset | same (§8) | aligned |
| Energy window | 10 ms-Leq only (rejects reflections) | **+ LIAeq,100ms** full free-field decay (§7) | deliberate |
| Peak/impulse window | short, clipped to reject reverberant reflections | **100 ms, equal to the energy window** (§2.8) | deliberate |
| Averaging | linear Pa/Pa·ms mean → dB | same (§10) | aligned |
| A-weighting | `adsgn.m` (IEC 1672) | same prototype (§9) | aligned |
