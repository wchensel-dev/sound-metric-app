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

# Shot onset: first raw-pressure sample above this level (Pa). The onset
# threshold is now recorded per shot (``Shot.trigger_pa``, defaulting to
# DEFAULT_TRIGGER_PA below) and passed into ``find_onset`` at marking. This
# constant is the *legacy / unknown-trigger fallback*: a shot with no recorded
# trigger (``trigger_pa`` is None — every row marked before the field existed)
# is analysed at 1 Pa, matching TBAC's ``find(Y>1.)`` and the historical
# behaviour, so nothing already computed shifts.
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

# Nominal DewesoftX acquisition standard: a hardware trigger (historically 1 Pa,
# later raised by the techs to 10 Pa and now 2 Pa to reject wind gusts) with a
# 10 ms pre-trigger lead and 200 ms post-trigger capture (T = 210 ms, N = 42 000
# at fs = 200 kHz). The recorder's trigger level is not stored in the file; the
# app re-derives onset from the data and records the trigger used per shot
# (``Shot.trigger_pa``, default DEFAULT_TRIGGER_PA). These are the config
# source-of-record constants behind MATH.md §1's `fs`/`N`/`T` rows and §2.3; the
# formulas use each file's actual fs and N, so of this set only CAPTURE_MS is
# read at runtime — the "no onset" warning cites it as the expected frame length.
EXPECTED_FS: float = 200_000.0
LEAD_MS: float = 10.0
POST_MS: float = 200.0
CAPTURE_MS: float = LEAD_MS + POST_MS  # 210 ms nominal frame
EXPECTED_SAMPLES: int = 42_000  # CAPTURE_MS at EXPECTED_FS

# Pre-trigger floor: how many samples from the very start of the capture are
# averaged into the per-channel baseline diagnostic (see
# `metrics.pretrigger_floor_pa`). 100 samples is 0.5 ms at EXPECTED_FS, well
# inside the LEAD_MS pre-trigger lead, so the mean is taken from quiet
# pre-shot signal on a nominal capture. Purely a QC readout: no metric is
# corrected by it, and nothing in the analysis path reads it.
PRETRIGGER_FLOOR_SAMPLES: int = 100

# Exponential RMS time-weighting constants for SPL-over-time display, IEC 61672.
# "Fast" and "Slow" are the standard sound-level-meter time constants; they turn
# the per-cycle swing of the raw waveform into a continuous level envelope.
FAST_TIME_S: float = 0.125
SLOW_TIME_S: float = 1.0

# Default local SQLite database file (relative to working dir).
DEFAULT_DB_PATH: str = "sound_metrics.db"


# --------------------------------------------------------------------------- #
# Batch-average targets
# --------------------------------------------------------------------------- #
# How many included shots of each role a batch aims to average, per mic
# position. These are **soft targets**, not hard caps: the store never refuses an
# inclusion that overshoots them, and reports show progress against them so the
# user can see when a batch is short or over. Regulars come from several
# clusters (a 3-shot cluster contributes two, a 4-shot cluster three), which is
# why inclusion is tracked per shot rather than per cluster — whole clusters
# cannot cleanly land on exactly 5.

TARGET_FRP_SHOTS: int = 3
TARGET_REGULAR_SHOTS: int = 5


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

#: Settings key holding the default onset trigger threshold (Pa) pre-filled when
#: marking a shot. See :func:`get_default_trigger_pa`.
TRIGGER_PA_KEY: str = "default_trigger_pa"

#: Default onset trigger threshold (Pa) for a freshly marked shot. Matches the
#: recorder's current 2 Pa hardware trigger; overridable per shot at marking and
#: globally via :func:`set_default_trigger_pa`. See ``ONSET_THRESHOLD_PA`` for
#: the separate legacy fallback used when a shot records no trigger at all.
DEFAULT_TRIGGER_PA: float = 2.0

#: Ammo presets seeded for a fresh install (no ammo definitions saved yet).
DEFAULT_AMMO_DEFINITIONS: tuple[str, ...] = (
    "LC M193 (5.56)",
    "LC M855 (5.56)",
    "PMC Bronze (5.56)",
    "Black Hills 77gr OTM (5.56)",
    "Winchester 147gr T&P (308)",
    "LC M118LR (308)",
    "Syntech TM 147gr (9mm)",
    "Federal 115gr Champion FMJ RN (9mm)",
    "Hornady Black 208gr (300 BLK)",
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


def get_default_trigger_pa() -> float:
    """The configured default onset trigger (Pa), falling back to :data:`DEFAULT_TRIGGER_PA`.

    Pre-filled into the mark form / CLI so a freshly marked shot records the
    recorder's current trigger unless the operator overrides it. A stored value
    that is not a positive number is rejected rather than silently accepted, so a
    corrupt setting surfaces instead of quietly skewing every new onset.
    """
    settings = load_settings()
    if TRIGGER_PA_KEY not in settings:
        return DEFAULT_TRIGGER_PA
    stored = settings[TRIGGER_PA_KEY]
    try:
        value = float(stored)
    except (TypeError, ValueError):
        raise ValueError(
            f"Setting {TRIGGER_PA_KEY!r} must be a number, got {type(stored).__name__}."
        )
    if not value > 0.0:
        raise ValueError(f"Setting {TRIGGER_PA_KEY!r} must be positive, got {value}.")
    return value


def set_default_trigger_pa(value: float) -> float:
    """Persist the default onset trigger (Pa) and return it. Must be positive."""
    trigger = float(value)
    if not trigger > 0.0:
        raise ValueError(f"Trigger threshold must be positive, got {trigger}.")
    settings = load_settings()
    settings[TRIGGER_PA_KEY] = trigger
    save_settings(settings)
    return trigger
