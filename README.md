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

The exact formulas, constants, and assumptions behind every metric are
documented in [MATH.md](MATH.md). **Keep it in sync:** any change to the math
models in `src/sound_metric_app/dsp/` (or the constants in `config.py`) must be
reflected in `MATH.md` in the same change.

## Data Model & Workflow

The app is organized around a pipeline that turns raw capture files into
metric-annotated shots, files them in a containment tree, and reports per-batch
averages over the shots the user brings forward.

> **Design principle:** state transitions are **user-actuated** wherever
> reasonable. The app suggests and computes; the user commits. Ingesting,
> marking, bringing shots forward, and closing batches are explicit user actions
> rather than automatic triggers.

### Containment tree

```
SKU
└─ Platform                (test rig, e.g. AR15 16" barrel)
   └─ Ammo                 (e.g. 5.56 M855)
      └─ Batch             (one test session: date, typical weather, notes)
         └─ Cluster        (one string of fire, shots kept in order)
            └─ Shot        (one gunshot event)
               ├─ Channel: muzzle_left    (AI 1 / transducer A)
               └─ Channel: shooters_ear   (AI 2 / transducer B)
```

**SKU, Platform, and Ammo are the three test conditions.** Every SKU holds many
platforms, every platform many ammo types, and a specific SKU + Platform + Ammo
path is a **test combination**. Batches live under that combination. (In storage
the three collapse into one `combinations` row, since they are only ever
meaningful together.)

- A **Batch** is **one test session** — the day's date, the weather typical for
  it, and free-form notes. It is *not* a SKU.
- A **Cluster** is **one string of fire**, its shots kept in order.
- A **Shot** is a single gunshot event, captured as **one file**.
- **Muzzle vs ear sits INSIDE the shot as two channels, not as a folder level.**
  Same shot, same timestamp, order, and cluster, diverging only at averaging
  time. This keeps the two transducers living together as one shot while still
  letting each be averaged separately. The channels map directly to the DAQ
  inputs: **AI 1 = muzzle left, AI 2 = shooter's ear**.
- **Metrics are computed per mic channel**, so every shot yields a muzzle-left
  result and a shooter's-ear result from its single file.

### What drives the roll-up (separate from nesting)

The tree above is **pure containment**. Three per-shot pieces of state do the
organizing:

- **`shot_order` within its cluster.** Order 0 is the **FRP** (first round pop),
  everything after is **regular**. Role is **derived** from order, so every
  cluster has exactly one FRP by construction and re-ordering a shot can never
  leave a stale role behind.
- **`position`,** which lives on the **channel** (muzzle_left / shooters_ear),
  not the shot.
- **`included` flag,** the thing that moves a shot from the data bank into the
  batch average. Idle by default, flipped on when brought forward, and where an
  exclusion reason (high winds, ambient noise) is recorded.

### Inclusion, and hitting 3 FRP / 5 regular exactly

The `included` flag lives on the **shot** as the source of truth, with a **bring
cluster forward** action that sets the flag on that cluster's shots. Regulars
come from multiple clusters (a 1,2,3 contributes two regulars, a 1,2,3,4
contributes three), so bringing whole clusters forward can't cleanly land on
exactly 5. Shot-level inclusion lets you pull **3 FRPs and 5 regulars as two
independent counts**, while still allowing a 50-cluster, 100-shot batch where
only selected shots feed the average. The 3 and 5 are **soft targets, not hard
caps** — the app reports progress against them and never refuses an overshoot.

### Two views

**Data bank view** — every cluster and shot, included or idle. The complete
archive; **nothing is deleted when left out of an average.**

**Batch average view** — the filter where `included` is true, grouped by position
× role, producing **four output slots per batch**:

```
muzzle_left  · FRP
muzzle_left  · regular
shooters_ear · FRP
shooters_ear · regular
```

The 3-FRP / 5-regular target applies **per position**, so each channel averages
the same underlying selected shots on its own axis.

### Single-shot data flow

1. **Raw ingest** — shot arrives with its own data (date, time, Pascals, sample
   rate) on both channels (AI 1 muzzle left, AI 2 shooter's ear).
2. **Processing** — manual tags added, specific weather for that shot, plus
   `shot_order` and cluster assignment.
3. **Data bank** — shot sits as part of a cluster, **idle by default**.
4. **Batch management** — clusters/shots reviewed; selected ones brought forward
   (`included = true`). Unselected shots sit idle, optionally with an exclusion
   reason.
5. **Batch averaging** — included shots roll up into the four output slots.

### 1. Ingestion — the input folder

- Raw capture files land in an **input folder** (drop target for Dewesoft
  exports).
- On ingest, each file becomes an **Unmarked Data Set** — a raw capture the app
  knows about but has no test context for yet.
- One file = one shot, containing both mic streams.

#### Filename convention

Capture files follow a fixed, app-controlled naming scheme:

```
<suppressor_sku>_<test_platform>_<cluster>_<shot_order>.dxd

  suppressor_sku   suppressor being tested, e.g. SUP-1234   → combination key
  test_platform    firearm / fixture, e.g. AR15             → combination key
  cluster          1-based string-of-fire number, e.g. 02   → which cluster
  shot_order       0-based shot number, e.g. 0003           → order within it; 0 = FRP
```

Example:

```
SUP-1234_AR15_02_0000.dxd   ← the FRP of cluster 2
SUP-1234_AR15_02_0001.dxd   ← the second round of that string
```

Encoding the cluster in the filename fixes each string of fire at capture time,
so a shot arrives already knowing which cluster it belongs to, its position
within it, and therefore **whether it is the FRP**. Ammo is tagged during
marking, not encoded in the name.

**The shot number is 0-based** because DewesoftX numbers its exports from zero —
the first capture of a string comes out as `0000`, the next as `0001`. That
trailing `0000` is what identifies the FRP, so the app takes Dewesoft's counter
as-is rather than making you renumber every file. The cluster field is ours, not
Dewesoft's, and stays **1-based**: a cluster of `0` is still rejected.

### 2. Marking — shot information

The user annotates each Unmarked Data Set, turning it into a **marked shot** with
the metadata below. Channel tagging is **auto-filled from the DAQ convention**
(AI 1 → Muzzle Left, AI 2 → Shooter's Ear) and stays **overridable** so a capture
that breaks the convention can still be tagged by hand.

| Field | Level | Notes |
|---|---|---|
| **Suppressor SKU** | combination | Suppressor under test |
| **Test Platform** | combination | Firearm / test fixture |
| **Ammo** | combination | Ammunition / load identifier — not in the filename |
| **Cluster** | shot | Which string of fire, seeded from the filename |
| **Shot Order** | shot | Position within its cluster, 0-based as Dewesoft exports it; **0 = FRP**, rest regular |
| **Role** | shot | **Derived** from Shot Order — never entered by hand |
| **Wind Speed** | shot | This shot's *specific* condition — recorded **per shot** |
| **Temp** | shot | Ambient temperature — recorded **per shot** |
| **Relative Humidity** | shot | Ambient RH — recorded **per shot** |
| **Captured** | shot | When the shot was fired — read from the capture file's Dewesoft `start_store_time` at marking (read-only) |
| **Mic channel → ML / SE** | channel | Which stream is which mic (auto-tagged) |

Environmental fields are captured on each individual shot, not once per batch —
conditions drift between shots. The **batch** separately carries the session's
*typical* weather, its date, and notes.

A freshly marked shot lands in the data bank **idle**: marking never sets
`included`, and re-marking never changes it.

Metrics (Peak dB, Peak dBA, Peak Impulse, LIAeq,100ms) are computed **per mic
channel** — each stream gets its own values from the DSP pipeline.

### 3. Placement — combinations, sessions, and strings of fire

- A shot is placed into its **combination** (SKU + Platform + Ammo), that
  combination's **open batch**, and the **cluster** its filename names.
- The user **closes** a batch to define the session; once closed, further testing
  on that combination starts a **new batch** rather than reopening the old one.
- Cluster indices are scoped to their batch: cluster 1 of one session and cluster
  1 of the next are different strings of fire.

### 4. Aggregation — the four output slots

Averages are computed **per batch**, over its **included** shots only, split by
mic position × derived role into the four slots above. Positions are never mixed
and roles are never mixed. Averaging is done in the linear domain and converted
to dB once (see [MATH.md](MATH.md) §10).

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
  services/    headless workflow: ingest, mark, cluster, aggregate
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

# Desktop GUI (workflow app)
python -m sound_metric_app.ui.main_window
```

The desktop app is the GUI counterpart to the `sma` CLI below — same services,
same local database. It opens four tabs matching the workflow:

- **Ingest** — shows the configured input folder, an **Ingest** button, and the
  list of Unmarked Data Sets with their parsed cluster, shot order, and derived
  role (plus a summary of malformed/unreadable files). Select a shot and click
  **Mark selected shot →** to jump to marking.
- **Mark** — pick an unmarked shot; its **Muzzle Left**/**Shooter's Ear**
  channels arrive pre-tagged from the AI 1 / AI 2 convention and stay editable.
  Fill in ammo + metadata (the **Role** field echoes FRP/Regular live as you type
  a shot order) and click **Mark** to compute and store its metrics. The shot
  lands in the data bank **idle**.
- **Data bank** — the Combination → Batch → Cluster → Shot tree: every shot,
  included or idle. Each shot row carries an **inclusion checkbox**; **Bring
  forward** / **Set idle…** act on the selected shot *or* a whole cluster, and
  setting a shot idle prompts for an exclusion reason. Each batch row shows its
  progress (`FRP: 2/3   Regular: 5/5`). **Close batch** ends the session. To fix
  an entry, select a node and click **Edit…** (or double-click): editing a
  **batch** opens its session form (label, date, typical weather, notes);
  editing a **shot** reopens the marking form pre-filled with where it actually
  landed, and saving re-marks it — re-placing it in the corrected
  combination/batch/cluster and recomputing its metrics. Inclusion is never
  changed by an edit.
- **Batch average** — the four position × role output slots for the selected
  batch, over its included shots only. Empty slots are shown as *none included*
  rather than hidden, and each populated slot expands into the shots behind it.
  Clicking a metric cell on a shot row graphs it.

Ingest and mark run off the UI thread, so a large capture never freezes the
window; service errors surface as dialogs.

### Workflow CLI (`sma`)

The `sma` command drives the full ingest → mark → bring-forward → report pipeline
over a local SQLite database (`--db`, default `sound_metrics.db`).

```powershell
# One-time: set the input folder the ingest command scans by default
sma config set-input-folder C:\captures\inbox
sma config show

# Ingest new capture files as Unmarked Data Sets (uses the configured folder,
# or pass one explicitly). --no-validate skips the readability check.
sma ingest
sma ingest C:\captures\inbox

# See what's waiting, then mark a shot. Channels auto-tag from AI 1 / AI 2, so
# --se/--ml are only needed to override a non-conforming capture.
sma list unmarked
sma mark 1 --ammo M855 --wind-speed 5 --temp 72 --rh 40
sma mark 2 --ammo M855 --se "Mic B" --ml "Mic A"   # manual override

# Record the session's context on the batch
sma batch 1 --label "Morning string" --date 2026-07-22 --wind-speed 4 --temp 88 `
            --notes "clear, light crosswind"

# Browse the tree
sma list combinations
sma list batches
sma list clusters --batch 1

# Data bank: every cluster and shot, [x] included / [ ] idle
sma bank 1

# Bring shots forward — a whole string of fire, or one shot at a time to land on
# exactly 3 FRPs and 5 regulars. Excluding records why.
sma include cluster 1
sma include shot 5
sma exclude shot 4 --reason "high winds"

# The four position x role slots, then close the batch to define the session
sma report --batch 1
sma report --combination 1
sma close-batch 1
```

The legacy single-file analyzer (`sma-analyze`) is unchanged.

## Tests

```powershell
pytest
```

Point `SMA_SAMPLE_DXD` at a real file (or drop one in `data/`) to enable the
real-file ingestion tests.

The GUI tests run headless via Qt's offscreen platform:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"; pytest
```

## Validation TODO

The **Peak Impulse** and **LIAeq,100ms** definitions are provisional. To lock
them: open a known file in DewesoftX, read its four displayed values, and tune
`dsp/metrics.py` until they match. Then capture golden-file regression tests.
```
