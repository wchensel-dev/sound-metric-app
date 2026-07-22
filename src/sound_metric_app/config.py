"""Global constants and persisted app settings."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Reference sound pressure for dB SPL (20 micropascals).
P_REF: float = 20e-6

# --------------------------------------------------------------------------- #
# Acquisition & analysis windows
# --------------------------------------------------------------------------- #
# Every metric is computed over a fixed window anchored to the detected shot
# onset (first raw-pressure sample above ONSET_THRESHOLD_PA), aligning with
# TBAC's process_string.m. See MATH.md §2/§6/§7.

# Shot onset: first raw-pressure sample above this level (Pa). TBAC uses 1 Pa.
ONSET_THRESHOLD_PA: float = 1.0

# Peak/impulse search window after onset (ms): the signed peak and the
# positive-phase impulse are found within [onset, onset + PEAK_WINDOW_MS].
# Deliberately equal to LIAEQ_WINDOW_MS below: one 100 ms free-field decay
# window for every onset-anchored metric, so peak/impulse and LIAeq can never
# disagree about which events are in scope for a frame (MATH.md §3).
PEAK_WINDOW_MS: float = 100.0

# Peak 10 ms-Leq: rectangular running-Leq integration time (s), and the
# post-onset span its running maximum is searched over (ms).
LEQ_TAU_S: float = 0.010
LEQ_SEARCH_MS: float = 25.0

# Proprietary LIAeq: A-weighted equivalent level over the free-field energy
# window [onset, onset + LIAEQ_WINDOW_MS] (MATH.md §7).
LIAEQ_WINDOW_MS: float = 100.0

# Nominal DewesoftX acquisition standard: a 1 Pa trigger with a 10 ms
# pre-trigger lead and 200 ms post-trigger capture (T = 210 ms, N = 42 000 at
# fs = 200 kHz). These are the config source-of-record constants behind
# MATH.md §1's `fs`/`N`/`T` rows and §2.3; the formulas use each file's actual
# fs and N, so of this set only CAPTURE_MS is read at runtime — the "no onset"
# warning cites it as the expected frame length.
EXPECTED_FS: float = 200_000.0
LEAD_MS: float = 10.0
POST_MS: float = 200.0
CAPTURE_MS: float = LEAD_MS + POST_MS  # 210 ms nominal frame
EXPECTED_SAMPLES: int = 42_000  # CAPTURE_MS at EXPECTED_FS

# Exponential RMS time-weighting constants for SPL-over-time display, IEC 61672.
# "Fast" and "Slow" are the standard sound-level-meter time constants; they turn
# the per-cycle swing of the raw waveform into a continuous level envelope.
FAST_TIME_S: float = 0.125
SLOW_TIME_S: float = 1.0

# Default local SQLite database file (relative to working dir).
DEFAULT_DB_PATH: str = "sound_metrics.db"


# --------------------------------------------------------------------------- #
# Persisted app settings
# --------------------------------------------------------------------------- #
#
# A small JSON file holds cross-invocation settings the CLI/GUI need — notably
# the configured *input folder* the ``ingest`` command scans by default. The
# file location resolves to the ``SMA_CONFIG`` environment variable if set, else
# a ``sma_config.json`` in the working directory (mirroring ``DEFAULT_DB_PATH``,
# which is likewise working-directory relative).

DEFAULT_CONFIG_PATH: str = "sma_config.json"

#: Settings key holding the configured input folder for ``ingest``.
INPUT_FOLDER_KEY: str = "input_folder"

#: Settings key holding the user's ammo definitions — the ammo types offered as
#: presets when marking a shot. See :func:`get_ammo_definitions`.
AMMO_DEFINITIONS_KEY: str = "ammo_definitions"

#: Ammo presets seeded for a fresh install (no ammo definitions saved yet).
DEFAULT_AMMO_DEFINITIONS: tuple[str, ...] = (
    "LC M193 (5.56)",
    "LC M855 (5.56)",
    "Black Hills 77gr OTM (5.56)",
)


def config_path() -> Path:
    """Location of the settings file (``$SMA_CONFIG`` or the cwd default)."""
    return Path(os.environ.get("SMA_CONFIG", DEFAULT_CONFIG_PATH))


def load_settings() -> dict:
    """Read the settings file into a dict; an absent file yields ``{}``.

    Raises ``ValueError`` if the file exists but cannot be read or parsed, so a
    corrupt settings file surfaces rather than silently reverting to defaults.
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read settings file {path}: {exc}") from exc


def save_settings(settings: dict) -> None:
    """Write ``settings`` to the settings file as pretty-printed JSON."""
    config_path().write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def get_input_folder() -> str | None:
    """The configured input folder, or ``None`` if none has been set."""
    return load_settings().get(INPUT_FOLDER_KEY)


def set_input_folder(folder: str | os.PathLike) -> Path:
    """Persist the input folder (stored as a resolved absolute path) and return it."""
    resolved = Path(folder).resolve()
    settings = load_settings()
    settings[INPUT_FOLDER_KEY] = str(resolved)
    save_settings(settings)
    return resolved


def _normalize_ammo(definitions: list[str]) -> list[str]:
    """Strip, drop blanks, and de-duplicate ``definitions``, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in definitions:
        name = raw.strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def get_ammo_definitions() -> list[str]:
    """The configured ammo presets, falling back to :data:`DEFAULT_AMMO_DEFINITIONS`.

    A fresh install (no ammo key saved) yields the built-in defaults so the mark
    form always offers something. Once the user saves their own list — even an
    empty one — that list is honoured verbatim.
    """
    settings = load_settings()
    if AMMO_DEFINITIONS_KEY not in settings:
        return list(DEFAULT_AMMO_DEFINITIONS)
    stored = settings[AMMO_DEFINITIONS_KEY]
    if not isinstance(stored, list):
        raise ValueError(
            f"Setting {AMMO_DEFINITIONS_KEY!r} must be a list of strings, "
            f"got {type(stored).__name__}."
        )
    return _normalize_ammo([str(item) for item in stored])


def set_ammo_definitions(definitions: list[str]) -> list[str]:
    """Persist the ammo presets (normalized) and return the stored list."""
    normalized = _normalize_ammo(list(definitions))
    settings = load_settings()
    settings[AMMO_DEFINITIONS_KEY] = normalized
    save_settings(settings)
    return normalized
