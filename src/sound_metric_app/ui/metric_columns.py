"""The metric and diagnostic columns the report views spend, named once.

Two tabs read this list and must offer the same metrics under the same names:
the Batch average tree spends them as columns, the Compare tab's picker spends
them as dropdown rows. Keeping the pair here means neither can grow a metric the
other has not heard of.
"""

from __future__ import annotations

#: The report's metric columns, in display order: (header label, stored metric
#: key). Shared by the Batch average tree (which spends them as columns) and the
#: Compare tab's metric picker (which spends them as dropdown rows), so the two
#: can only ever offer the same metrics under the same names. Every key is one
#: :func:`~sound_metric_app.dsp.build_metric_trace` accepts.
_REPORT_METRICS = (
    ("Peak Pa", "peak_pa"),
    ("Peak dB", "peak_db"),
    ("Peak dBA", "peak_dba"),
    ("Impulse Pa·ms", "impulse_pa_ms"),
    ("Impulse dB·ms", "peak_impulse_db"),
    ("Peak Leq10ms dBA", "leq10ms_db"),
    ("LIAeq,100ms dBA", "liaeq_100ms_db"),
)

#: Header for the pre-trigger floor wherever it is shown. Named once because two
#: trees carry the column with different bodies — Batch average one mic per row,
#: the data bank both mics in one cell — and a label that drifted between them
#: would read as two different diagnostics.
_PRETRIGGER_FLOOR_LABEL = "Pre-trig floor Pa"

#: Per-shot diagnostic columns: (header label, stored key). Deliberately *not*
#: part of :data:`_REPORT_METRICS` — every key there must be one
#: :func:`~sound_metric_app.dsp.build_metric_trace` accepts (the Compare picker
#: and the click-to-graph handler both assume it), and these have no curve. They
#: are QC readouts on the capture, not measurements of the shot, so they trail
#: the metrics and only shot rows carry them. Named for the report the way
#: :data:`_REPORT_METRICS` is, distinct from the storage layer's
#: ``_DIAGNOSTIC_COLUMNS``, which lists database column names.
_REPORT_DIAGNOSTICS = (
    (_PRETRIGGER_FLOOR_LABEL, "pretrigger_floor_pa"),
)
