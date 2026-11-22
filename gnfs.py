import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def read_networks(path, frame):
    with np.load(path, allow_pickle=False) as archive:
        weights = archive["causal"]
        starts = archive["window_start_sample"]
        ends = archive["window_end_sample"]
        variables = archive["variables"]
        timestamps = archive["window_end_timestamp"]
        values_digest = archive["scada_values_sha256"].item() if "scada_values_sha256" in archive else None

    if not np.array_equal(variables, frame.columns.to_numpy(dtype=str)):
        raise ValueError("The network variables and SCADA columns differ or are in a different order.")
    count = len(variables)
    if count < 2 or weights.ndim != 3 or weights.shape[1:] != (count, count) or len(weights) < 2:
        raise ValueError("GNFS needs at least two square directed networks with at least two variables.")
    if not np.isfinite(weights).all() or (weights < 0).any() or np.diagonal(weights, axis1=1, axis2=2).any():
        raise ValueError("IEE weights must be finite and nonnegative with a zero diagonal.")
    if starts.shape != (len(weights),) or ends.shape != starts.shape:
        raise ValueError("The number of window indices differs from the number of networks.")
    if not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(ends.dtype, np.integer):
        raise ValueError("Window indices must be integers.")
    if (starts < 1).any() or (ends < starts).any() or (ends > len(frame)).any():
        raise ValueError("Window indices fall outside the supplied SCADA records.")
    if (np.diff(starts) <= 0).any() or (ends - starts != ends[0] - starts[0]).any():
        raise ValueError("Network windows must appear in time order and have the same length.")
    actual_timestamps = np.array([time.isoformat() for time in frame.index[ends - 1]])
    if not np.array_equal(timestamps, actual_timestamps):
        raise ValueError("Network endpoint timestamps differ from the supplied SCADA timestamps.")
    if values_digest is not None:
        from preprocess_scada import sample_digest

        if values_digest != sample_digest(frame.to_numpy()):
            raise ValueError("SCADA measurements differ from those used to construct the networks.")
    return weights, starts, ends, timestamps


def local_network_entropy(weights):
    undirected = np.maximum(weights, weights.swapaxes(1, 2))
    total_strength = undirected.sum(axis=2)
    degree = (undirected > 0).sum(axis=2)
    proportions = np.divide(
        undirected, total_strength[:, :, None], out=np.zeros_like(undirected),
        where=total_strength[:, :, None] > 0,
    )
    log_proportions = np.log(proportions, out=np.zeros_like(proportions), where=proportions > 0)
    entropy = -(proportions * log_proportions).sum(axis=2)
    return np.divide(entropy, degree, out=np.zeros_like(total_strength), where=degree > 0)


def score_components(samples, weights, starts, ends):
    entropy = local_network_entropy(weights)
    mad = np.empty_like(entropy)
    for index, (start, end) in enumerate(zip(starts, ends)):
        window = samples[start - 1:end]
        mad[index] = np.abs(window - window.mean(axis=0)).mean(axis=0)

    edge_change = np.abs(np.diff(weights, axis=0))
    incident_change = edge_change.sum(axis=1) + edge_change.sum(axis=2)
    ncsf = incident_change / (2 * (weights.shape[1] - 1))
    return {
        "LNE": entropy,
        "MAD": mad,
        "U": np.abs(np.diff(entropy, axis=0)),
        "R": ncsf,
        "V": np.abs(np.diff(mad, axis=0)),
    }


def fit_rms(components):
    lne_change = components["U"]
    ncsf = components["R"]
    return {
        "c_U": float(np.sqrt(np.mean(lne_change * lne_change))),
        "c_R": float(np.sqrt(np.mean(ncsf * ncsf))),
    }


def calculate_scores(components, calibration, variables, top_nodes):
    c_u = float(calibration["c_U"])
    c_r = float(calibration["c_R"])
    if not np.isfinite([c_u, c_r]).all() or c_u <= 0 or c_r <= 0:
        raise ValueError("Healthy-reference RMS coefficients must be finite and strictly positive.")
    if not 1 <= top_nodes <= len(variables):
        raise ValueError("Top-node count must be between 1 and the number of variables.")

    network_change = components["U"] / c_u + components["R"] / c_r
    node_values = 0.5 * network_change * components["V"]
    window_ids = pd.Index(np.arange(2, len(node_values) + 2), name="window")
    nodes = pd.DataFrame(node_values, columns=variables, index=window_ids).sort_index(axis=1)
    gnfs = nodes.apply(lambda row: row.nlargest(top_nodes).mean(), axis=1)
    return nodes, gnfs


def confirmed_warning_mask(scores, threshold):
    exceedances = np.asarray(scores) > threshold
    recent_counts = np.convolve(exceedances.astype(int), np.ones(5, dtype=int), mode="full")[:len(exceedances)]
    return exceedances & (recent_counts >= 3)


def main():
    from preprocess_scada import read_scada

    parser = argparse.ArgumentParser(description="Calibrate or calculate the global network fluctuation score.")
    parser.add_argument("operation", choices=["calibrate", "score"])
    parser.add_argument("scada", type=Path, help="Prepared SCADA data used to construct the networks.")
    parser.add_argument("networks", type=Path, help="NPZ file produced by mic_iee.py.")
    parser.add_argument("output", type=Path, help="New directory for the GNFS results.")
    parser.add_argument("--calibration", type=Path, help="Healthy-reference calibration JSON for score.")
    parser.add_argument("--top-nodes", type=int, default=12)
    args = parser.parse_args()
    if (args.operation == "score") != (args.calibration is not None):
        parser.error("Supply --calibration when scoring a segment.")
    if args.output.exists():
        parser.error("The output directory already exists.")

    frame = read_scada(args.scada)
    weights, starts, ends, timestamps = read_networks(args.networks, frame)
    components = score_components(frame.to_numpy(), weights, starts, ends)
    if args.operation == "calibrate":
        calibration = fit_rms(components)
    else:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
    nodes, gnfs = calculate_scores(components, calibration, frame.columns, args.top_nodes)
    global_scores = pd.DataFrame({
        "window_k1": gnfs.index.to_numpy(),
        "timestamp_UTC": timestamps[1:],
        "score": gnfs.to_numpy(),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        result = Path(temporary) / "result"
        result.mkdir()
        global_scores.to_csv(result / "global_scores.csv", index=False, encoding="utf-8-sig")
        nodes.to_csv(result / "node_scores.csv", encoding="utf-8-sig")
        np.savez_compressed(
            result / "score_components.npz",
            variables=frame.columns.to_numpy(dtype=str),
            **components,
        )
        if args.operation == "calibrate":
            calibration["healthy_max_GNFS"] = float(gnfs.max())
            (result / "calibration.json").write_text(
                json.dumps(calibration, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
        result.rename(args.output)

    print(f"Wrote {len(gnfs)} GNFS scores to {args.output}.")


if __name__ == "__main__":
    main()
