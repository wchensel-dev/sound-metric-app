"""Global constants and configuration."""

from __future__ import annotations

# Reference sound pressure for dB SPL (20 micropascals).
P_REF: float = 20e-6

# Analysis window (LIAeq is reported per 100 ms).
WINDOW_MS: float = 100.0

# Nominal DewesoftX acquisition parameters (used for validation warnings only).
EXPECTED_FS: float = 200_000.0
EXPECTED_SAMPLES: int = 20_000

# Impulse ("I") time-weighting constants, IEC 61672.
IMPULSE_RISE_S: float = 0.035
IMPULSE_FALL_S: float = 1.5

# Default local SQLite database file (relative to working dir).
DEFAULT_DB_PATH: str = "sound_metrics.db"
