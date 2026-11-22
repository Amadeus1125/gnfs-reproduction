# GNFS Reproduction

This example starts from preprocessed SCADA data. The two inputs in `t01_example/` each contain 500 timestamped records and 24 monitoring variables. `healthy.csv` is the T01 healthy reference used for calibration, and `fault.csv` is the T01 transformer cooling system fault segment.

## Data Source and License

The T01 inputs are adapted from EDP Open Data's [Wind Farm 1 wind turbine SCADA signals for 2017](https://edp.com/en/innovation/wind-farm-1-wind-turbine-scada-signals-2017). The example uses selected records and monitoring variables, standardized measurements, and time interpolation for gaps in the fault segment. The T01 example data and derived data outputs are shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Implementation

1. **MIC screening.** The script computes the maximal information coefficient (MIC) in each 100-record window, advancing one record at a time. It uses the MIC_e estimator with a grid bound of 8 and `c = 15`. It takes the median of the upper-triangular off-diagonal entries and retains variable pairs at or above that value.
2. **IEE estimation.** For each retained pair, interventional embedding entropy (IEE) is estimated separately in both directions using differences between neighboring states in delay embedding space. The defaults are a delay length of 2, a local neighborhood of 10, and 4 neighbors for mutual information estimation. A pair is removed when both directed IEE weights are below `1e-12`.
3. **LNE calculation.** Local network entropy (LNE) uses the larger of the two directed IEE weights as the undirected edge weight. For each node, the positive incident weights are normalized, and their entropy is divided by the number of connected neighbors. Isolated nodes have LNE of zero.
4. **GNFS calculation.** Between adjacent windows, `U` is the absolute change in LNE, `R` is the sum of absolute changes in the incoming and outgoing IEE weights divided by `2(p - 1)`, and `V` is the absolute change in mean absolute deviation about the window mean. Here `p` is the number of variables. The healthy reference supplies the RMS coefficients `c_U` and `c_R`. The node score is `0.5 * (U / c_U + R / c_R) * V`, and GNFS is the mean of the 12 highest node scores.

## Run the T01 Example

Use a Python environment with NumPy, pandas, SciPy, scikit-learn, Numba, and minepy. Run these commands from the directory containing this README. A fresh output directory keeps the included example results available for comparison.

```sh
mkdir t01_run
python mic_iee.py t01_example/healthy.csv t01_run/healthy_networks.npz
python mic_iee.py t01_example/fault.csv t01_run/fault_networks.npz
python gnfs.py calibrate t01_example/healthy.csv t01_run/healthy_networks.npz t01_run/healthy_result
python gnfs.py score t01_example/fault.csv t01_run/fault_networks.npz t01_run/fault_result --calibration t01_run/healthy_result/calibration.json
```

| Output | Contents |
| --- | --- |
| `healthy_networks.npz`, `fault_networks.npz` | MIC matrices, MIC thresholds and screening masks, directed IEE matrices, variable names, window indices, and window-end timestamps |
| `healthy_result/calibration.json` | Healthy-reference RMS coefficients and maximum healthy GNFS |
| `global_scores.csv` in each result directory | GNFS, window number, and window-end timestamp |
| `node_scores.csv` in each result directory | Scores for all monitoring variables in each score window |
| `score_components.npz` in each result directory | LNE, MAD, and the three GNFS components `U`, `R`, and `V` |

Each input produces 401 networks and 400 GNFS values. The first GNFS is assigned to the end of window 2. The reference run yields `c_U = 0.020518810621869946`, `c_R = 0.018369927901480863`, and a maximum healthy GNFS of approximately `0.01343254`. With the fixed warning threshold of `0.015`, the first confirmed warning occurs at window 149. The fault logged at `2017-08-11T13:14:00Z` maps to the SCADA sample at `2017-08-11T13:20:00Z` (window 282), giving a warning lead time of 22 hours and 10 minutes.

## Optional Outputs

`plot_gnfs.py` can produce a PNG, PDF, or SVG score plot from `global_scores.csv`. It requires Matplotlib.

```sh
python plot_gnfs.py t01_run/fault_result/global_scores.csv t01_run/t01_gnfs.png --threshold 0.015 --fault-window 282 --dpi 1200
```

`noise_experiment.py` runs the trials listed in a supplied CSV and uses the same healthy-reference calibration JSON. It requires joblib. The trial CSV columns are `case,input,fault_time_utc,noise_type,noise_level,seed`. Each row specifies one trial, with `gaussian` or `impulsive` as the noise type. Relative `input` paths are resolved from the trial CSV directory. The output includes one `noisy_scada.csv` and one `global_scores.csv` per trial, plus `trials.csv` and `summary.csv` for lead times and missed warnings.

```sh
python noise_experiment.py trials.csv t01_noise_run --calibration t01_run/healthy_result/calibration.json
```
