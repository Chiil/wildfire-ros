# wildfire_ros

Interactive notebook for Rothermel (1972) surface-fire **rate of spread (ROS)**
using the Scott & Burgan (2005) 40 standard fire behavior fuel models (FBFM40).

## Use

```bash
.venv/bin/python -m jupyterlab        # or: code rothermel_ros.ipynb
```

Open **`rothermel_ros.ipynb`**, run all cells, then:

1. pick a fuel model from the dropdown (GR1 … SB4),
2. set dead/live fuel moisture, wind, and slope,
3. choose how wind is given (10-m, 20-ft, or midflame) and a **wind adjustment
   factor** model (unsheltered, sheltered-under-canopy, or direct),
4. read the ROS (m/min, km/h, m/s, ft/min, chains/h) plus the intermediate
   Rothermel parameters and the applied midflame wind, which update live.

The WAF converts 10-m/20-ft wind to the **midflame** wind Rothermel's model
needs (Albini & Baughman 1979; Andrews 2012, RMRS-GTR-266). `wind_adjustment_factor`,
`waf_unsheltered`, and `waf_sheltered` are also importable for direct use.

Computation is in Rothermel's original imperial units (so intermediate
parameters match published tables); ROS is converted to metric at the end.

## Files

| file | purpose |
|---|---|
| `rothermel.py` | fuel-model loader + Rothermel/Albini ROS (`compute_ros`) |
| `rothermel_ros.ipynb` | interactive ipywidgets notebook |
| `data/fbfm40.json` | the 40 standard fuel models |
| `validate.py` | reproduces published characteristic SAV / packing ratio for all 40 |
| `build_notebook.py` | regenerates the notebook |

## Validation

```bash
.venv/bin/python validate.py
```

Reproduces the published characteristic SAV, packing ratio, and relative packing
ratio for all 40 models. Four transcription errors in the source JSON are
corrected in `rothermel.py` (`_CORRECTIONS`): SH1/SH6/TL3 100-h loads and the
SH9 fuel-bed depth (3.0 → 4.4 ft).

## Programmatic use

```python
from rothermel import load_fuel_models, compute_ros, Moisture, mps_to_ft_min, deg_to_slope_fraction
M = load_fuel_models()
r = compute_ros(M["GR2"], Moisture.from_percent(6, 7, 8, 60, 90),
                mps_to_ft_min(5), deg_to_slope_fraction(10))
print(r.ros_m_min, "m/min")
```

## References

- Rothermel 1972, INT-115 · Albini 1976, INT-GTR-30
- Scott & Burgan 2005, RMRS-GTR-153 · Andrews 2018, RMRS-GTR-371

> Note: the fuel-model set is **Scott & Burgan** (often mis-cited as "Burgess").
