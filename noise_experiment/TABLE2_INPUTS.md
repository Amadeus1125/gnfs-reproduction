# Table 2 noise experiments

`table2_trials.csv` lists the 180 realizations summarized in Table 2. It contains the case, relative input path, logged fault time in UTC, noise type, noise strength, and seed required by the experiment script.

The source is EDP Wind Farm 1's 2017 SCADA dataset. The input grids contain 500 records at 10-minute intervals. All ranges below are inclusive and in UTC. [Dataset and contact information](../DATA_ACCESS.md)

| Case | EDP observation range | Experiment input range | Input file |
| --- | --- | --- | --- |
| T01 transformer cooling system fault | 2017-08-08 17:10 to 2017-08-12 10:30 | 2017-08-08 22:00 to 2017-08-12 09:10 | `../t01_example/fault.csv` |
| T07 hydraulic group oil leakage fault | 2017-06-14 11:40 to 2017-06-18 01:10 | 2017-06-14 14:00 to 2017-06-18 01:10 | `T07_20170617.csv` |
| T06 gearbox fault | 2017-10-14 05:40 to 2017-10-17 16:50 | 2017-10-14 05:40 to 2017-10-17 16:50 | `T06_20171017.csv` |

The timestamp column is followed by the 24 measurement columns listed below in order.

- **1–8.** Mean generator speed, generator bearing temperature, generator phase temperatures A/B/C, hydraulic oil temperature, gearbox oil temperature, gearbox bearing temperature.
- **9–16.** High-voltage transformer phase temperatures A/B/C, phase A inverter temperature, generator slip-ring temperature, rotor inverter temperatures 1/2/3.
- **17–24.** Grid active power, grid phase voltages 1/2/3, grid phase currents 1/2/3, second generator bearing temperature.

T06 z-score parameters are fitted to the 500 listed observations using `StandardScaler`. Exact replay for T07 requires the matching prepared input on the specified grid. For help with this input, contact the corresponding author listed in [Data Access](../DATA_ACCESS.md).

With the T07 and T06 CSVs in this directory, run the following command from the repository root.

```sh
python noise_experiment.py noise_experiment/table2_trials.csv table2_run --calibration t01_example/healthy_result/calibration.json
```

`table2_run/summary.csv` contains the lead-time mean and sample standard deviation in hours for each case, noise type, and strength. Sum missed warnings over the three strengths to obtain the last column of Table 2.
