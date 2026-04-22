# Examples

Reference implementations showing how to fill the template for different domains. Each example is a **minimal** working fill — just enough to demonstrate the pattern.

## How to read these

Each subfolder specializes the template's slots for one domain. Use them as reference when filling the real domain on finale day.

| Example | Domain | What it shows |
|---|---|---|
| `echo/` | Simplest possible | Bare-minimum Action/Observation/reward — the "hello world" |
| *(add more as time permits)* | — | — |

## Files per example

Each example should contain, at minimum:

- `models.py` — specialized `ActionCommand` enum, `TaskSpec` fields
- `scenarios.py` — 1-2 scenarios
- `rules.py` — 2-3 rules
- `rewards.py` — component weights with rationale

**Do NOT** copy these into the main template — they're reference only. The template stays domain-agnostic until Round 2 theme drops.

## What this folder is NOT for

- Not the place for the Round 2 specialization — that replaces the top-level template files
- Not a library/framework — examples are read-only references
- Not part of the Docker build or the deployed HF Space
