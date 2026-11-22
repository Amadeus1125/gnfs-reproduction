import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from gnfs import calculate_scores, confirmed_warning_mask, score_components
from mic_iee import compute_network_window
from preprocess_scada import read_scada


WINDOW_LENGTH = 100
TOP_NODES = 12
IMPULSE_PROBABILITY = 0.01
NOISE_TYPES = ("gaussian", "impulsive")
TRIAL_COLUMNS = ("case", "input", "fault_time_utc", "noise_type", "noise_level", "seed")


def read_trials(path):
    trials = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not set(TRIAL_COLUMNS).issubset(rows.fieldnames or []):
            raise ValueError(f"Required trial CSV columns: {', '.join(TRIAL_COLUMNS)}.")
        for line, row in enumerate(rows, start=2):
            try:
                case, source, fault, kind, raw_level, raw_seed = (
                    (row[field] or "").strip() for field in TRIAL_COLUMNS
                )
                level = float(raw_level)
                seed = int(raw_seed)
                fault_time = pd.to_datetime(fault, utc=True)
                if not case or not source or pd.isna(fault_time):
                    raise ValueError("case, input and fault_time_utc are required")
                if kind not in NOISE_TYPES or not np.isfinite(level) or level <= 0 or seed < 0:
                    raise ValueError("use gaussian or impulsive, a positive noise level, and a nonnegative integer seed")
                input_path = (path.parent / source).resolve(strict=True)
                if not input_path.is_file():
                    raise ValueError("input must be a SCADA file")
            except (ValueError, OSError) as error:
                raise ValueError(f"Invalid trial on CSV line {line}: {error}.") from error
            trials.append((case, input_path, fault_time, kind, level, seed))
    if not trials:
        raise ValueError("The trial CSV must contain at least one trial.")
    return trials


def add_noise(values, kind, level, seed):
    channel_std = values.std(axis=0, ddof=0)
    if not np.isfinite(channel_std).all() or (channel_std <= 0).any():
        raise ValueError("Every clean monitoring variable must have a finite positive standard deviation.")
    target_std = level * channel_std
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        noise = rng.standard_normal(values.shape) * target_std
    elif kind == "impulsive":
        pulses = rng.random(values.shape) < IMPULSE_PROBABILITY
        pulse_std = target_std / np.sqrt(IMPULSE_PROBABILITY)
        noise = np.where(pulses, rng.standard_normal(values.shape) * pulse_std, 0.0)
    else:
        raise ValueError(f"Unknown noise type: {kind}.")
    noisy = values + noise
    if not np.isfinite(noisy).all():
        raise ValueError("Noise produced non-finite monitoring values.")
    return np.asfortranarray(noisy)


def run_trial(number, trial, calibration, threshold, output):
    case, input_path, fault_time, kind, level, seed = trial
    clean = read_scada(input_path)
    if len(clean) <= WINDOW_LENGTH or len(clean.columns) < TOP_NODES:
        raise ValueError(f"{case}: need at least {WINDOW_LENGTH + 1} samples and {TOP_NODES} variables.")
    if not clean.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all():
        raise ValueError(f"{case}: supply prepared SCADA records on a consecutive 10-minute grid.")
    if not clean.index[0] <= fault_time <= clean.index[-1]:
        raise ValueError(f"{case}: the logged fault time falls outside the SCADA input.")

    fault_position = int(clean.index.searchsorted(fault_time, side="left"))
    mapped_fault_time = clean.index[fault_position]
    mapped_fault_window = fault_position + 2 - WINDOW_LENGTH
    noisy = add_noise(clean.to_numpy(), kind, level, seed)
    starts = np.arange(1, len(noisy) - WINDOW_LENGTH + 2)
    ends = starts + WINDOW_LENGTH - 1
    weights = np.stack([
        compute_network_window(noisy[start - 1:end])[1]
        for start, end in zip(starts, ends)
    ])
    components = score_components(noisy, weights, starts, ends)
    _, scores = calculate_scores(components, calibration, clean.columns, TOP_NODES)
    confirmed = confirmed_warning_mask(scores.to_numpy(), threshold)
    score_times = clean.index[ends[1:] - 1]
    first_indices = np.flatnonzero(confirmed)
    first = int(first_indices[0]) if len(first_indices) else None
    warning_time = score_times[first] if first is not None else None
    lead_time = None
    if warning_time is not None and warning_time < mapped_fault_time:
        lead_time = (mapped_fault_time - warning_time).total_seconds() / 3600

    trial_id = f"trial_{number:04d}"
    destination = output / trial_id
    destination.mkdir()
    noisy_frame = pd.DataFrame(noisy, columns=clean.columns)
    noisy_frame.insert(0, "Timestamp", [time.isoformat() for time in clean.index])
    noisy_frame.to_csv(
        destination / "noisy_scada.csv", index=False, encoding="utf-8-sig", float_format="%.17g",
    )
    score_frame = pd.DataFrame({
        "window_k1": scores.index.to_numpy(),
        "timestamp_UTC": [time.isoformat() for time in score_times],
        "score": scores.to_numpy(),
        "threshold": threshold,
        "exceeds_threshold": scores.to_numpy() > threshold,
        "confirmed_3of5": confirmed,
    })
    score_frame.to_csv(destination / "global_scores.csv", index=False, encoding="utf-8-sig")
    return {
        "trial": trial_id,
        "case": case,
        "input": str(input_path),
        "fault_time_utc": fault_time.isoformat(),
        "fault_mapped_time_utc": mapped_fault_time.isoformat(),
        "fault_window": mapped_fault_window if mapped_fault_window > 0 else None,
        "noise_type": kind,
        "noise_level": level,
        "seed": seed,
        "first_warning_window": int(scores.index[first]) if first is not None else None,
        "first_warning_time_utc": warning_time.isoformat() if warning_time is not None else None,
        "lead_time_hours": lead_time,
        "missed_warning": int(lead_time is None),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the noise trials specified in an input CSV.")
    parser.add_argument("trials", type=Path, help="CSV with case, input, fault_time_utc, noise_type, noise_level, seed.")
    parser.add_argument("output", type=Path, help="New directory for the trial data and results.")
    parser.add_argument("--calibration", required=True, type=Path, help="GNFS healthy-reference calibration JSON.")
    parser.add_argument("--threshold", type=float, default=0.015)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    if not np.isfinite(args.threshold) or args.threshold <= 0 or args.jobs < 1:
        parser.error("The threshold must be finite and positive and jobs must be at least 1.")
    if args.output.exists():
        parser.error("The output directory already exists.")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
    scales = np.array([float(calibration["c_U"]), float(calibration["c_R"])])
    if not np.isfinite(scales).all() or (scales <= 0).any():
        parser.error("The calibration coefficients must be finite and positive.")
    trials = read_trials(args.trials)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            records = Parallel(n_jobs=args.jobs)(
                delayed(run_trial)(number, trial, calibration, args.threshold, staged)
                for number, trial in enumerate(trials, start=1)
            )
        results = pd.DataFrame(records)
        for column in ("fault_window", "first_warning_window"):
            results[column] = results[column].astype("Int64")
        results["lead_time_hours"] = pd.to_numeric(results["lead_time_hours"])
        results.to_csv(staged / "trials.csv", index=False, encoding="utf-8-sig")
        groups = ["case", "input", "fault_time_utc", "noise_type", "noise_level"]
        summary = results.groupby(groups, sort=False, as_index=False).agg(
            trials=("trial", "size"),
            detected=("lead_time_hours", "count"),
            missed_warnings=("missed_warning", "sum"),
            mean_lead_time_hours=("lead_time_hours", "mean"),
            sd_lead_time_hours=("lead_time_hours", "std"),
        )
        summary.to_csv(staged / "summary.csv", index=False, encoding="utf-8-sig")
        staged.rename(args.output)

    print(f"Wrote {len(trials)} noise trials to {args.output}.")


if __name__ == "__main__":
    main()
