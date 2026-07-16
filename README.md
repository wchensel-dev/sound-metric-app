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

## Data Model & Workflow

The app is organized around a pipeline that turns raw capture files into
metric-annotated shots, organizes them by suppressor SKU into batches (and by
ammo + platform into groups), and reports per-group averages.

> **Design principle:** state transitions are **user-actuated** wherever
> reasonable. The app suggests and computes; the user commits. Ingesting,
> marking, and closing batches are explicit user actions rather than automatic
> triggers.

### Hierarchy

The **Suppressor SKU** is the top of the organizational hierarchy — a batch *is*
a SKU under test. One SKU is tested across **multiple ammo and test platforms**,
so those live as a grouping level beneath it.

```
Batch  ──►  Group             ──►  Shot            ──►  Mic channels
(SKU)       (Ammo + Platform)      (one capture file)   (SE + MR)
```

- A **Batch** is one **Suppressor SKU**. It collects every shot fired to test
  that suppressor, across all ammo and platforms.
- A **Group** is the set of shots within a batch that share the same **Ammo +
  Test Platform**. Groups are the meaningful unit for averaging (identical test
  conditions). Shots are ordered by **Shot Order** within a group.
- A **Shot** is a single firing event, captured as **one file**. That file
  carries **two mic channels** — **Shooter's Ear (SE)** and **Muzzle Right
  (MR)** — recorded simultaneously.
- **Metrics are computed per mic channel**, so every shot yields an SE result
  and an MR result from its single file.

### 1. Ingestion — the input folder

- Raw capture files land in an **input folder** (drop target for Dewesoft
  exports).
- On ingest, each file becomes an **Unmarked Data Set** — a raw capture the app
  knows about but has no test context for yet.
- One file = one shot, containing both the SE and MR mic streams.

#### Filename convention

Capture files follow a fixed, app-controlled naming scheme:

```
<suppressor_sku>_<test_platform>_<shot_order>.dxd

  suppressor_sku   suppressor being tested, e.g. SUP-1234   → batch key
  test_platform    firearm / fixture, e.g. AR15             → group key (with ammo)
  shot_order       zero-padded shot number, e.g. 003        → orders shots in the group
```

Example:

```
SUP-1234_AR15_003.dxd
```

- **Batch key** = `suppressor_sku`.
- **Group key** = `test_platform` + **Ammo** (Ammo is tagged during marking, not
  encoded in the filename).
- `shot_order` seeds **Shot Order** (user-editable after marking).

### 2. Marking — shot information

The user annotates each Unmarked Data Set, turning it into a **marked shot** with
the metadata below. The user also **tags which mic channel is SE and which is
MR** within the file — this channel tagging is **user-defined for now**, until we
find a clean way to detect it automatically.

| Field | Level | Notes |
|---|---|---|
| **Suppressor SKU** | batch | Suppressor under test — the batch key |
| **Test Platform** | group | Firearm / test fixture — part of the group key |
| **Ammo** | group | Ammunition / load identifier — part of the group key |
| **Shot Order** | shot | Sequence position within its group |
| **Wind Speed** | shot | Environmental condition — recorded **per shot** |
| **Temp** | shot | Ambient temperature — recorded **per shot** |
| **Relative Humidity** | shot | Ambient RH — recorded **per shot** |
| **Mic channel → SE / MR** | channel | Which stream in the file is which mic |

Environmental fields (Wind Speed, Temp, Relative Humidity) are captured on each
individual shot, not once per batch — conditions can drift between shots.

Metrics (Peak dB, Peak dBA, Peak Impulse, LIAeq,100ms) are computed **per mic
channel** — the SE and MR streams each get their own values from the DSP
pipeline.

### 3. Clustering — batches and groups

- Shots are clustered into a **batch** by **Suppressor SKU**, then into
  **groups** by **Test Platform + Ammo** within that batch.
- The user **closes** a batch to define it; once closed, further testing of a
  similar type starts a **new batch** rather than reopening the old one.

### 4. Aggregation — averages

Averages are computed **per group** (matched Suppressor SKU + Test Platform +
Ammo) and **separately for each mic position** — each group yields a parallel set
of SE averages and MR averages, so neither positions nor test conditions are
mixed.

### Storage

Processed shots, batches, and averages are persisted to **local SQLite** for
now. A **remote store** is planned so data can be accessed from anywhere by any
client — but the current scope is **local-only** to keep things simple.

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
