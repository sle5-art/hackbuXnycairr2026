# Model card: SensorGap

*Format follows Mitchell et al. (2019), "Model Cards for Model Reporting." Numbers marked **[prelim]** come from one clean night (Nov 7–8, 2026) and are re-run with `python -m sensorgap.validate --strict` before the demo. Smoke-day figures come from a replay of June 7, 2023 (`python -m app.compare`).*

## 1. Model details
- **What it is:** a spatial **Gaussian process (GP) regression** that estimates hourly PM2.5 (fine particles, µg/m³) across New York City on a 0.5 km grid and reports a **calibrated 95% uncertainty band** for every grid cell. Two decision layers sit on top: an **AQI label rule** that abstains when the band is too wide, and a **greedy sensor-placement** ranking.
- **Built by:** Naman Kukreti and teammate, NYCAIRR 2026 hackathon (HackBU, Track 4: Environment), November 7–8, 2026. Version 0.1, research prototype.
- **Implementation:** Python, scikit-learn `GaussianProcessRegressor`. Kernel `Constant × Matérn(ν = 1.5)` on x/y in kilometres, length scale bounded to 1–100 km. Separate measurement noise per sensor type, chosen each hour by maximum marginal likelihood from {0.1, 0.25, 0.5, 1, 2, 4} µg/m³, with low-cost noise never below reference noise. Refit on every hourly snapshot. CPU only, a few seconds per fit.
- **Code:** github.com/sle5-art/hackbuXnycairr2026 (`sensorgap/model.py`, `validate.py`, `labels.py`, `placement.py`).

## 2. Intended use
- **Primary use:** an **advisory** public map that shows where PM2.5 estimates are well supported and where they are guesses, plus a **ranked shortlist** of candidate low-cost sensor sites for residents, advocates and city staff to review.
- **Primary users:** NYC residents (especially people with asthma), community groups, and agency planners.
- **Out of scope:** regulatory or compliance decisions (attainment, enforcement, permits); health or medical advice for individuals; forecasting future hours; indoor air; any location outside the NYC bounding box; using a single hour's value as an official AQI.

## 3. Factors
- **Sensor type:** reference monitors (city DOHMH, state DEC via EPA AirNow) vs corrected low-cost PurpleAir sensors. The model treats them differently through learned noise.
- **Geography and local sources:** errors concentrate near local sources (highways, tunnels, bus terminals) and at the edges of the network (Staten Island, New Jersey). The model's band width does grow with distance to the nearest sensor (correlation ~0.81 [prelim]), but actual error does **not**: misses come from local hotspots.
- **Air-quality regime:** validated on one clean night (~3 µg/m³). A monitors-only replay of the June 7, 2023 wildfire-smoke day (202–460 µg/m³) shows very wide bands and mostly withheld labels; the live configuration is **not validated** for smoke days (sections 7, 9).
- **Neighbourhood equity:** low-cost sensor density may track income. This is **being measured, not assumed**; the result will be reported either way.

## 4. Metrics
- **Coverage** of the 50/68/80/90/95/99% bands on held-out reference monitors (target: coverage ≈ claimed level).
- **Calibration factor (`sd_scale`):** how much the bands must widen so 95% of held-out readings fall inside; never below 1.0.
- **Spearman rank correlation** between absolute error and (a) the model's predicted sd, (b) distance to the nearest training sensor.
- **Placement:** weighted variance removed per pick; overlap between the asthma-weighted and population-only rankings; band width before/after for the highest-asthma quarter of tracts vs the rest.

## 5. Training data
- **PurpleAir** outdoor sensors in the NYC bounding box (~86 returned, ~65 kept per hour). Readings with channel A/B disagreement above 5 µg/m³ **and** 61% are dropped (Barkjohn et al., 2021); the rest are corrected with EPA's US-wide equation `0.524 × raw − 0.0862 × RH + 5.75`.
- **NYC DOHMH** real-time monitors (~14 mapped sites) and **EPA AirNow** state monitors (~15). Timestamps converted to UTC.
- About 94 readings per hourly snapshot. **All feeds are preliminary and unvalidated by their providers.** Raw API responses are kept in `data/raw/` for reproducibility.

## 6. Evaluation data and method
- **Leave-one-monitor-out with a 1 km buffer:** each reference monitor and every sensor within 1 km of it is hidden, the GP is refit on the rest and predicts the monitor. The hidden reading's own noise is added to the predictive sd.
- **Strict mode:** noise levels are re-learned inside every test without the hidden monitor, so nothing leaks.
- Snapshots taken within 30 minutes of each other count as one hour.
- **Time-split calibration:** the widening factor is fit on the earlier half of snapshots and tested on the later half.

## 7. Quantitative results [prelim: 3 hourly snapshots, 87 held-out readings]

| Measure | Result |
|---|---|
| Inside the 95% band (strict) | **90%** (target 95%) |
| Inside the 99% band (strict) | 95% |
| Calibration factor `sd_scale` | ×1.22 |
| Band half-width after calibration | ~±1.2 µg/m³ near sensors to ~±1.7 far |
| Learned noise (reference / low-cost) | ~1.0 / ~2.0 µg/m³; length scale ~2.4 km |
| Spearman: abs. error vs GP sd / vs distance to nearest sensor | −0.14 / −0.14 (neither predicts error well) |
| Mean error by source | city monitors +0.66, state monitors −0.52 µg/m³ vs map |
| Largest miss | Midtown West DOHMH (near Lincoln Tunnel): 7–9 observed vs ~3 predicted, every hour |
| Placement overlap, asthma vs population | 2 of 10 sites |
| Band width, highest-asthma quarter of tracts | 1.26 → 1.19 µg/m³ after 10 sites (~6% cut; ~3% elsewhere) |

**Smoke-day replay** (22 monitors, no PurpleAir; wider noise range used in the replay only):

| | Tonight | June 7, 2023 |
|---|---|---|
| Readings | 0–11 µg/m³ | 202–460 µg/m³ |
| 95% band | ±1 | ±52–104 |
| Cells with a confident label | 100% | 6% |
| Hidden monitors inside band | 97% | 100% |

None of the 10 suggested sensor sites match between the two days.

**Reading these numbers:** the uncertainty bands are close to honest but slightly overconfident before calibration, and the misses come from local hotspots the smooth model cannot see. We do **not** claim the GP's uncertainty predicts error better than distance to the nearest sensor.

## 8. Ethical considerations
- **Abstention over false confidence:** an AQI category is shown only when the whole 95% band lies inside it; otherwise the map says "X to Y, not enough data." A gap in coverage is never displayed as clean air.
- **Weights are value choices:** placement weights by estimated adults with asthma (CDC PLACES tract model, adult population × prevalence). Because only 2 of 10 picks match a population-only ranking, the weighting changes the answer; both rankings are shown so people can see the trade-off.
- **Secondary modeled data:** CDC PLACES asthma rates are themselves estimates, not measurements.
- **Privacy:** only public, aggregate sensor and tract data are used; no personal data is collected.
- **Risk of misuse:** a calm, confident-looking map could be read as official. The dashboard carries a preliminary-data footer and this card states it is advisory only.

## 9. Caveats and recommendations
- **Location-only model:** no wind, traffic, buildings or emission inventories, so estimates between sensors are rough. One stationary kernel for the whole city.
- **Small sample, one clean night.** Results will move; re-run strict validation before relying on any number.
- **High-pollution events need their own validation.** The live noise range cannot fit smoke-day data; the replay needed a wider range, and the live model with that range has not been re-validated. On the day it matters most, the honest map withholds most labels.
- **Siting depends on conditions:** no suggested site matched between the clean night and the smoke day, so placement should be reviewed across several kinds of days before anyone installs sensors.
- **Hourly values are concentrations, not AQI**; categories use a NowCast-style 12-hour average.
- **The equity-of-coverage claim stands only if** the income check shows a skew in NYC low-cost sensor density.
- **Recommendations:** treat the map as a screening tool; prioritise checking flagged hotspots and "not enough data" areas with additional monitoring; review placement recommendations with community input before installing sensors.
