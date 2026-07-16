---
name: add-mr-suggestion
description: Evaluate a merge request review suggestion, implement if valid, and provide a commit message. Usage: /add-mr-suggestion "<suggestion text>"
disable-model-invocation: true
argument-hint: "<paste the MR suggestion here>"
---

# Address MR Suggestion

You are evaluating a merge request review suggestion. Follow this process exactly.

## Input

The suggestion to evaluate is provided after the slash command invocation:

$ARGUMENTS

## Step 1: Evaluate Validity

Before touching any code, reason through whether the suggestion is worth implementing:

- **Correctness**: Does it fix a real bug or prevent a real problem? For DSP code, does it preserve the correctness of the acoustic metrics (Peak dB, Peak dBA, Peak Impulse, LIAeq100ms)?
- **Architecture fit**: Does it respect the layer boundaries in `src/sound_metric_app/` (`ingestion` → `dsp` → `storage` → `ui`/`cli`)? Does it align with patterns already used in this codebase?
- **Scope**: Is it in scope for the change being reviewed, or is it unrelated cleanup?
- **Trade-offs**: Does it introduce complexity that outweighs the benefit?
- **Project conventions**: Does it respect the layout (source under `src/sound_metric_app`, tests mirroring modules under `tests/`) and stay within the declared dependencies in `pyproject.toml`?

State your verdict clearly:

- **Valid** — explain why, then implement it
- **Invalid** — explain why, then stop (do not implement)
- **Needs clarification** — explain what's ambiguous and ask before proceeding

## Step 2: Implement (if valid)

Read the affected file(s) before making any changes. Then:

1. Make the smallest change that addresses the suggestion
2. Do not refactor unrelated code
3. Keep each layer's responsibilities intact — don't put DSP logic in the UI/CLI, or storage concerns in the DSP layer
4. Run `ruff check` (and `ruff format` if formatting changed) and fix any lint errors
5. Run the test suite with `pytest -q` and fix any failures; add or update tests under `tests/` when behavior changes

## Step 3: Report

Summarize:

- What you changed and why
- Any files touched
- Anything you chose NOT to change and why

## Step 4: Provide Commit Message

Give short commit message suggestion
