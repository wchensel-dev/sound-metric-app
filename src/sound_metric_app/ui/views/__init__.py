"""The five tabs of the workflow window, one module each.

Five views over the same Phase B services the ``sma`` CLI drives, wired through
:class:`~sound_metric_app.ui.controller.WorkflowController`:

1. **Ingest / Unmarked** (:mod:`.ingest`) — scan the input folder, list Unmarked
   Data Sets.
2. **Mark** (:mod:`.marking`) — annotate a shot, confirm its ML/SE channel tags,
   compute + store metrics. The shot lands in the data bank idle.
3. **Data bank** (:mod:`.data_bank`) — the Combination -> Batch -> Cluster -> Shot
   tree: every shot the app has seen, included or idle, with the bring-forward
   actions that decide which ones feed an average, plus session editing and
   Close batch.
4. **Batch average** (:mod:`.batch_average`) — the four position x role output
   slots per batch (muzzle-left / shooter's-ear crossed with FRP / regular),
   positions and roles never mixed, each averaged row ending in a Copy button
   that yields the SilencerScout ``SSR1`` paste string for that slot.
5. **Compare** (:mod:`.compare`) — one metric's curve for any number of shots
   overlaid on a single graph, pinned there by a Compare button on a Batch
   average shot row or a Data bank shot row — the former one button per slot's
   mic, the latter one button per marked channel, since a data-bank row is not
   scoped to a position. Either source can reach an idle shot's curve just as
   well as an included one. Shots from different batches, SKUs, and mics all
   coexist here, and pinning the same shot/mic from both tabs is a no-op the
   second time (:class:`~sound_metric_app.ui.compare_series.CompareSeries` keys
   on shot id + position, not on which tab sent it) — the batch-average and
   data-bank tabs are where shots are chosen, this one is where they are read
   against each other.

The split between tabs 3 and 4 is the directive's two views: the data bank is the
complete archive where nothing is deleted for being left out, and the batch
average is the filter over ``included``.

Ingest, mark, include, and close are explicit buttons (README user-actuated
principle). The two file-reading operations (ingest, mark) run on a worker thread
so a large capture never freezes the window (see :mod:`.base`); every service
error surfaces as a dialog.
"""

from __future__ import annotations

from .base import _Task, _View
from .batch_average import BatchAverageView
from .compare import CompareView
from .data_bank import DataBankView
from .ingest import IngestView
from .marking import MarkingView

__all__ = [
    "_Task",
    "_View",
    "BatchAverageView",
    "CompareView",
    "DataBankView",
    "IngestView",
    "MarkingView",
]
