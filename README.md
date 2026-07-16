# Sound Metric App

Data-production app that ingests DewesoftX sound data (Pascals) and computes
impulse-noise acoustic metrics, with local storage and data-management tools.

## Metrics

| Metric | Status | Definition |
|---|---|---|
| **Peak dB** | stable | `20·log10(\|p\|_max / 20µPa)` |
| **Peak dBA** | stable | Peak of the IEC-61672 A-weighted signal |
| **Peak Impulse** | provisional | Max of Impulse ("I") time-weighted level — *pending Dewesoft validation* |
| **LIAeq,100ms** | provisional | A-weighted Leq over the 100 ms frame — *pending Dewesoft validation* |

Reference pressure: 20 µPa. Input: one `.dxd`/`.d7d` = one 100 ms frame
(20,000 samples @ 200 kHz).

## Layout

```
src/sound_metric_app/
  ingestion/   dwdatareader -> Frame  (native lib bundled, no SDK needed)
  dsp/         weighting, metrics, MetricsProcessor  (pure, tested)
  storage/     SQLite results database
  ui/          PySide6 desktop app
  cli.py       command-line analyzer
tests/         unit + real-file validation
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gui,dev]"
```

## Usage

```powershell
# List channels in a file
sma-analyze path\to\file.dxd --list

# Compute metrics (auto-detects the Pa channel)
sma-analyze path\to\file.dxd

# Compute and store in a local SQLite DB
sma-analyze path\to\file.dxd --store sound_metrics.db

# Desktop GUI
python -m sound_metric_app.ui.main_window
```

## Tests

```powershell
pytest
```

Point `SMA_SAMPLE_DXD` at a real file (or drop one in `data/`) to enable the
real-file ingestion tests.

## Validation TODO

The **Peak Impulse** and **LIAeq,100ms** definitions are provisional. To lock
them: open a known file in DewesoftX, read its four displayed values, and tune
`dsp/metrics.py` until they match. Then capture golden-file regression tests.
```
