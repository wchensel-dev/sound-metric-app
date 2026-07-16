# Build Plan — Sound Metric App

A numbered, dependency-ordered workflow for turning the current single-file DSP
analyzer into the full workflow app described in [README.md](README.md).

**How to use this in a future session:** point Claude at a task number, e.g.
*"Execute Task 4 from BUILD_PLAN.md."* Each task is self-contained: it states a
goal, what to build, which files it touches, its dependencies, and the
acceptance criteria that mean it's done. Do tasks in order unless a task says
otherwise — later tasks assume earlier ones exist.

---

## Where we're starting from (baseline)

Already built and working:

- **DSP core** (`dsp/`) — `MetricsProcessor` computes the 4 metrics for one
  channel of one frame. Pure and tested. **Keep as-is.**
- **Ingestion reader** (`ingestion/dewesoft_reader.py`) — reads *one*
  auto-detected Pa channel from a `.dxd`/`.d7d` into a `Frame`.
- **Models** (`models.py`) — `Frame`, `MetricResult`. Single-channel only.
- **Storage** (`storage/database.py`) — flat `results` table. **No** batch /
  group / shot hierarchy.
- **UI** (`ui/main_window.py`) — open one file, show 4 metrics. Placeholder.
- **CLI** (`cli.py`) — `sma-analyze <file>`.

The gap: everything in README §"Data Model & Workflow" (the
Batch→Group→Shot→Channel hierarchy, two-mic capture, filename parsing, and the
ingest→mark→cluster→aggregate pipeline) is not built yet.

---

## Phase A — Data foundation

### Task 1 — Domain models for the hierarchy
**Goal:** Introduce the Batch / Group / Shot / channel-metric data types and the
filename convention parser.
**Depends on:** none.
**Build:**
- In `models.py`, add dataclasses: `Batch` (SKU, closed flag), `Group`
  (test_platform, ammo), `Shot` (source_file, shot_order, wind_speed, temp,
  relative_humidity, SE/MR channel tags, marked flag), and a per-channel result
  type (extend or wrap `MetricResult` with `mic_position` = SE|MR).
- Add a `MicPosition` enum (`SE`, `MR`).
- Add `parse_capture_filename("SUP-1234_AR15_003.dxd") -> (sku, platform,
  shot_order)` with validation and a clear error on malformed names.
**Acceptance:**
- Unit tests parse valid names and reject malformed ones.
- Models importable; existing tests still pass (`pytest`).

### Task 2 — Two-channel (SE + MR) ingestion
**Goal:** Read *both* mic channels from a single file, not just one auto-picked
channel.
**Depends on:** Task 1.
**Build:**
- Add `read_capture(path) -> list[Frame]` (or a `ShotCapture` holding both
  frames) that returns every synchronous Pa channel in the file.
- Keep `read_frame` working for the CLI/back-compat path.
- Support user channel tagging: a way to say "channel X is SE, channel Y is MR"
  (README says channel→mic mapping is user-defined for now).
**Acceptance:**
- Given a two-channel file, ingestion yields two frames with distinct channel
  names.
- Real-file test guarded by `SMA_SAMPLE_DXD` passes when a sample is present.

### Task 3 — Storage schema for the hierarchy
**Goal:** Replace the flat `results` table with tables that persist the whole
model.
**Depends on:** Task 1.
**Build:**
- New schema: `batches`, `groups`, `shots`, `channel_metrics` (per-shot,
  per-mic), with foreign keys (batch→group→shot→metrics).
- Repository methods: create/close batch, upsert group, add unmarked shot, mark
  shot, save channel metrics, query shots-by-group, query group averages.
- Keep the old `results` table path available for the CLI, or migrate the CLI
  onto the new schema in Task 8.
**Acceptance:**
- A round-trip test creates a batch→group→shot→metrics and reads it back.
- DB is created fresh from schema on first open.

---

## Phase B — Workflow engine (headless services)

These are pure/logic services with no UI, so they're easy to test and reuse from
both CLI and GUI.

### Task 4 — Ingestion service (input folder → Unmarked Data Sets)
**Goal:** Turn the drop-target input folder into unmarked shots.
**Depends on:** Tasks 2, 3.
**Build:**
- A service that scans the configured input folder, parses each filename
  (Task 1), reads its channels (Task 2), and records each file as an **Unmarked
  Data Set** (a shot with provisional batch/group keys, `marked = false`).
- Idempotent: re-scanning doesn't duplicate already-ingested files.
- Ingest is a user-actuated action (README design principle) — expose it as an
  explicit call, not an auto-watcher (a watcher can be a later enhancement).
**Acceptance:**
- Dropping N files and running ingest yields N unmarked shots; running again adds
  0.

### Task 5 — Marking service (annotate → marked shot + metrics)
**Goal:** Apply user metadata to an unmarked set, tag SE/MR, and compute metrics.
**Depends on:** Tasks 3, 4.
**Build:**
- Accept marking metadata (SKU, test platform, ammo, shot order, wind speed,
  temp, RH) and the channel→SE/MR mapping for a shot.
- On mark: run `MetricsProcessor` per tagged channel, persist per-mic
  `channel_metrics`, set `marked = true`.
**Acceptance:**
- Marking an unmarked shot produces one SE and one MR metric row; the shot moves
  out of the unmarked list.

### Task 6 — Clustering & batch lifecycle
**Goal:** Group shots by SKU (batch) and platform+ammo (group); support closing
a batch.
**Depends on:** Task 5.
**Build:**
- Assign marked shots into the right batch (by SKU) and group (by platform +
  ammo); order shots by shot_order within a group.
- Batch **close** action: once closed, further similar testing starts a *new*
  batch (README §3). Guard against marking into a closed batch.
**Acceptance:**
- Shots land in the correct batch/group; a closed batch rejects new shots and a
  new batch is created instead.

### Task 7 — Aggregation service (per-group, per-mic averages)
**Goal:** Compute averages per group, separately for SE and MR.
**Depends on:** Task 6.
**Build:**
- For each group, average each of the 4 metrics across shots — **separately** for
  SE and MR (never mixed). Return a parallel SE-average / MR-average set per
  group.
**Acceptance:**
- A group with known shot values returns the expected SE and MR averages;
  positions are not mixed.

---

## Phase C — Interfaces

### Task 8 — CLI expansion over the services
**Goal:** Drive the whole pipeline from the command line.
**Depends on:** Tasks 4–7.
**Build:**
- Subcommands: `ingest` (scan input folder), `mark` (annotate a shot),
  `list` (unmarked / batches / groups), `close-batch`, `report` (group averages).
- Keep the existing `sma-analyze <file>` single-file path working.
**Acceptance:**
- A full ingest→mark→close→report cycle is runnable end-to-end from the CLI.

### Task 9 — Workflow UI
**Goal:** Replace the placeholder window with the workflow views.
**Depends on:** Tasks 4–7 (can start after 5; report view needs 7).
**Build:**
- Views: input-folder / unmarked-sets list with an **Ingest** action; a
  **Marking** form (metadata + SE/MR channel tagging); a **Batch → Group → Shot**
  tree with a **Close batch** action; a **group averages** report (SE vs MR).
- Honor the user-actuated principle: ingest, mark, and close are explicit
  buttons.
**Acceptance:**
- A user can complete ingest→mark→close→view-averages entirely in the GUI.

---

## Phase D — Validation & hardening

### Task 10 — Validate provisional metrics against DewesoftX
**Goal:** Lock down **Peak Impulse** and **LIAeq,100ms** (README "Validation
TODO").
**Depends on:** DSP core (already present); best after Task 2 for real files.
**Build:**
- Open a known file in DewesoftX, read its four displayed values, tune
  `dsp/metrics.py` until they match, then capture **golden-file regression
  tests**. Update the metric status in the README table from *provisional* to
  *stable* once matched.
**Acceptance:**
- Golden-file tests pin the tuned values; README metric table updated.

### Task 11 — End-to-end tests & packaging
**Goal:** Confidence and shippability.
**Depends on:** Tasks 8, 9.
**Build:**
- One end-to-end test exercising ingest→mark→cluster→aggregate against sample
  files. Verify `pip install -e ".[gui,dev]"` and the entry points. Refresh
  README usage if commands changed.
**Acceptance:**
- `pytest` green including the e2e test; documented commands run as written.

---

## Dependency map (quick reference)

```
1 ─┬─► 2 ─┐
   └─► 3 ─┴─► 4 ─► 5 ─► 6 ─► 7 ─┬─► 8 ─┐
                                └─► 9 ─┴─► 11
2 ─► 10  (independent; do when a real file is available)
```

## Deferred / out of current scope
- Remote store (README "Storage" — planned; current scope is local-only).
- Automatic SE/MR channel detection (user-defined for now).
- Automatic folder-watching ingest (current scope is user-actuated ingest).
