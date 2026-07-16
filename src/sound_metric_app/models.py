"""Core data models shared across ingestion, DSP, and storage layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Frame:
    """One acquisition frame of calibrated sound-pressure data, in Pascals.

    A DewesoftX ``.dxd`` capture for this application is a single 100 ms frame
    (20,000 samples at 200 kHz), so one file maps to one ``Frame``.
    """

    samples: np.ndarray  # 1-D float64, Pascals (calibration already applied)
    sample_rate: float  # Hz
    channel: str
    source_file: str
    timestamp: datetime | None = None
    events: list = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass
class MetricResult:
    """Computed acoustic metrics for a single :class:`Frame`."""

    peak_db: float
    peak_dba: float
    peak_impulse_db: float
    liaeq_100ms_db: float
    source_file: str
    channel: str
    sample_rate: float
    n_samples: int
    timestamp: datetime | None = None

    def as_row(self) -> dict:
        """Flat dict suitable for storage / CSV export."""
        return {
            "source_file": self.source_file,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "peak_db": self.peak_db,
            "peak_dba": self.peak_dba,
            "peak_impulse_db": self.peak_impulse_db,
            "liaeq_100ms_db": self.liaeq_100ms_db,
        }
