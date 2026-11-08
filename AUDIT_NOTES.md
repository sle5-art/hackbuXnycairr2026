# SensorGap audit notes (2026-11-08)

The full audit was run on 8 hours of real data (strict validation, 232 held-out readings, 29 monitors).
This file lists what was fixed in code and what is still open, with the line to say if a judge asks.

## Fixed in this branch

| Problem | Fix | Where |
|---|---|---|
| Map colours stretched to each hour's own min and max, so a 0.5 µg/m³ difference looked dramatic | Fixed 0-based scales: PM2.5 tops at 35.4 (top of Moderate) and steps up by AQI category only on bad days; "how unsure" tops at 5, then 10, 25... | `app/dashboard.py` `ramp()`, `fixed_top()`; test `test_colour_scale_is_fixed_not_stretched` |
| `validate --last N` overwrote `calibration.json` and `validation_summary.json` with a partial run | `--last` skips those two writes; its other files get a `_lastN` suffix | `sensorgap/validate.py` |
| PurpleAir reads ~1–2.8 µg/m³ below monitors within 1 km, and nothing measured it | `lowcost_offset()` measures the gap every hour; `correct_lowcost()` can shift PurpleAirs by it. **Off by default** (`CORRECT_LOWCOST_BIAS = False`) because strict validation got worse with it on: mean miss 1.29 → 1.37 µg/m³, 95% coverage 92.7% → 92.2%, calibration ×1.20 → ×1.27. Re-test with `python -m sensorgap.validate --strict --pa-offset` | `sensorgap/model.py`; test `test_purpleair_offset_is_measured_and_applied` |

## Still open (say it before a judge does)

- **The two official networks disagree.** NYC Health monitors read ~1.4 µg/m³ above NY State DEC monitors in NYC (8 hours). That is why the PurpleAir correction fails: there is no single truth to correct toward. On the 2023 smoke day, the two collocated Queens College monitors differed ~1.6×.
- **Learned noise is mostly equal** for monitors and corrected PurpleAirs (72% of fits). Don't claim cheap sensors get "half the trust".
- **The GP's uncertainty doesn't beat distance-to-sensor** at ranking misses (Spearman −0.11 vs −0.19; both weak). Misses come from local hotspots (Midtown West), not coverage gaps. Claim calibrated band width, not ranking.
- **Midtown West** (36061NY09929) is 7 of the 8 biggest misses; its 30.3 µg/m³ hour at 05:21 UTC is unverified. Without it, 96% coverage.
- **Placement changes with the hour** because the length scale is re-learned (tonight vs smoke day share 0 of 10 sites). A real deployment would pick the sites chosen most often over a month.
- **Asthma weighting is mostly adult headcount** (tract asthma 6.8–15.7%); 6 of 10 sites match population-only.
- **City feed lags ~2 h**, and the ingest loop froze once on 2026-11-08 06:21 UTC after the PurpleAir fetch (restarted; the request timeout doesn't cover DNS).

## Data for re-running validation

`data/static/history.csv` is a copy of the collected hourly snapshots (`data/snapshots/` is gitignored):

    python -m sensorgap.validate --strict --history data/static/history.csv
