# SensorGap model card

## Intended use
Advisory map of PM2.5 and its uncertainty for NYC; ranking candidate sensor sites. Not for regulatory decisions.

## Data
All feeds are preliminary and unvalidated. PurpleAir readings are corrected with EPA's US-wide equation.

## Validation
(to fill) Hold-out coverage of the 95% band on reference monitors; reliability curve; distance-to-sensor baseline.

## Known limits
- One stationary kernel for the whole city.
- AQI categories use a NowCast-style average; hourly values are concentrations, not AQI.
- Asthma prevalence (CDC PLACES) is itself a model estimate.
