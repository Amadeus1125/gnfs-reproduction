import argparse
from pathlib import Path

import numpy as np
from minepy import MINE
from numba import njit
from scipy.spatial import cKDTree
from scipy.special import psi


IEE_ZERO_TOLERANCE = 1e-12


def mic_matrix(series, grid_cells):
    count = len(series)
    scores = np.eye(count)
    estimator = MINE(alpha=grid_cells, c=15, est="mic_e")
    for source in range(count):
        for target in range(count):
            if source != target:
                estimator.compute_score(series[source], series[target])
                scores[source, target] = estimator.mic()
    return scores


def screen_pairs(mic, quantile):
    upper = np.triu_indices(len(mic), k=1)
    values = mic[upper]
    threshold = np.quantile(values, quantile)
    retained = np.zeros(mic.shape, dtype=bool)
    retained[upper] = values >= threshold
    return retained | retained.T, threshold


def delay_embeddings(series, delay_length):
    length = len(series)
    history = [series[delay_length - lag - 1:length - lag - 1] for lag in range(delay_length)]
    return np.column_stack(history), np.column_stack([series[delay_length:], *history])


@njit
def neighbor_counts(source, target, k):
    anchors, neighbors, source_dimension = source.shape
    target_dimension = target.shape[2]
    source_counts = np.zeros((anchors, neighbors), dtype=np.int64)
    target_counts = np.zeros((anchors, neighbors), dtype=np.int64)
    sizes = np.zeros(anchors, dtype=np.int64)

    for anchor in range(anchors):
        unique = np.empty(neighbors, dtype=np.int64)
        size = 0
        for candidate in range(neighbors):
            duplicate = False
            for previous in range(size):
                other = unique[previous]
                same = True
                for dimension in range(source_dimension):
                    if source[anchor, candidate, dimension] != source[anchor, other, dimension]:
                        same = False
                        break
                if same:
                    for dimension in range(target_dimension):
                        if target[anchor, candidate, dimension] != target[anchor, other, dimension]:
                            same = False
                            break
                if same:
                    duplicate = True
                    break
            if not duplicate:
                unique[size] = candidate
                size += 1
        sizes[anchor] = size
        if size <= k:
            continue

        source_distances = np.zeros((size, size))
        target_distances = np.zeros((size, size))
        for first in range(size):
            for second in range(first + 1, size):
                source_distance = 0.0
                target_distance = 0.0
                for dimension in range(source_dimension):
                    distance = source[anchor, unique[first], dimension] - source[anchor, unique[second], dimension]
                    source_distance = max(source_distance, abs(distance))
                for dimension in range(target_dimension):
                    distance = target[anchor, unique[first], dimension] - target[anchor, unique[second], dimension]
                    target_distance = max(target_distance, abs(distance))
                source_distances[first, second] = source_distances[second, first] = source_distance
                target_distances[first, second] = target_distances[second, first] = target_distance

        for first in range(size):
            distances = np.maximum(source_distances[first], target_distances[first])
            distances.sort()
            radius = distances[k]
            for second in range(size):
                if source_distances[first, second] < radius:
                    source_counts[anchor, first] += 1
                if target_distances[first, second] < radius:
                    target_counts[anchor, first] += 1

    return source_counts, target_counts, sizes


def directed_iee(series, retained, delay_length, neighbors, mi_neighbors, exclusion):
    anchors = series.shape[1] - delay_length
    embeddings = [delay_embeddings(variable, delay_length) for variable in series]
    sources = np.array([source for source, _ in embeddings])
    targets = np.array([target for _, target in embeddings])
    indices = np.arange(anchors)
    weights = np.zeros(retained.shape)

    for target in range(len(series)):
        retained_sources = np.flatnonzero(retained[:, target])
        if not len(retained_sources):
            continue
        target_history = targets[target]
        neighborhood = np.empty((anchors, neighbors), dtype=int)
        for time in range(anchors):
            allowed = indices[np.abs(indices - time) > exclusion]
            _, nearest = cKDTree(target_history[allowed]).query(
                target_history[time:time + 1], k=neighbors,
            )
            neighborhood[time] = allowed[nearest[0]]

        response = target_history[neighborhood] - target_history[:, None, :]
        for source in retained_sources:
            source_history = sources[source]
            perturbation = source_history[neighborhood] - source_history[:, None, :]
            source_counts, target_counts, sizes = neighbor_counts(perturbation, response, mi_neighbors)
            information = np.zeros(anchors)
            for size in np.unique(sizes):
                if size <= mi_neighbors:
                    continue
                selected = sizes == size
                source_mean = np.mean(psi(source_counts[selected, :size]), axis=1)
                target_mean = np.mean(psi(target_counts[selected, :size]), axis=1)
                information[selected] = np.maximum(0.0, psi(mi_neighbors) - source_mean - target_mean + psi(size))
            weights[source, target] = float(np.mean(information))

    return weights


def remove_near_zero_pairs(weights):
    below_tolerance = weights < IEE_ZERO_TOLERANCE
    return np.where(below_tolerance & below_tolerance.T, 0.0, weights)


def compute_network_window(samples, grid_cells=8, quantile=0.5, delay_length=2,
                           neighbors=10, mi_neighbors=4, theiler=2):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 2 or not np.isfinite(samples).all():
        raise ValueError("Supply a finite two-dimensional sample matrix with at least two variables.")
    if grid_cells < 4 or not 0 <= quantile <= 1:
        raise ValueError("Set MIC grid cells to at least 4 and the quantile within [0, 1].")
    if delay_length < 1 or theiler < 0 or not 2 <= mi_neighbors < neighbors:
        raise ValueError("Require positive delay length, nonnegative Theiler length, and 2 <= k < K.")
    exclusion = max(theiler, delay_length)
    if samples.shape[0] - delay_length - (2 * exclusion + 1) < neighbors:
        raise ValueError("The window is too short for the requested IEE neighborhood.")

    series = samples.T
    mic = mic_matrix(series, grid_cells)
    retained, threshold = screen_pairs(mic, quantile)
    causal = directed_iee(series, retained, delay_length, neighbors, mi_neighbors, exclusion)
    return mic, remove_near_zero_pairs(causal), retained, threshold


def main():
    from preprocess_scada import read_scada, sample_digest

    parser = argparse.ArgumentParser(description="Calculate sliding-window MIC and directed IEE matrices.")
    parser.add_argument("input", type=Path, help="Prepared SCADA data in CSV or XLSX format.")
    parser.add_argument("output", type=Path, help="Path for a new NPZ file.")
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--mic-grid-cells", type=int, default=8)
    parser.add_argument("--mic-quantile", type=float, default=0.5)
    parser.add_argument("--delay-length", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--mi-neighbors", type=int, default=4)
    parser.add_argument("--theiler", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("The output file already exists.")
    frame = read_scada(args.input)
    if not 1 <= args.window_length <= len(frame) or args.step < 1:
        parser.error("Window length must fit the input and step must be positive.")
    samples = frame.to_numpy()
    starts = np.arange(0, len(samples) - args.window_length + 1, args.step)
    shape = (len(starts), samples.shape[1], samples.shape[1])
    mic = np.empty(shape)
    causal = np.empty(shape)
    mic_mask = np.empty(shape, dtype=bool)
    thresholds = np.empty(len(starts))

    for index, start in enumerate(starts):
        window = samples[start:start + args.window_length]
        mic[index], causal[index], mic_mask[index], thresholds[index] = compute_network_window(
            window, args.mic_grid_cells, args.mic_quantile, args.delay_length,
            args.neighbors, args.mi_neighbors, args.theiler,
        )
    ends = starts + args.window_length - 1
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            variables=frame.columns.to_numpy(dtype=str),
            window_start_sample=starts + 1,
            window_end_sample=ends + 1,
            window_end_timestamp=np.array([value.isoformat() for value in frame.index[ends]]),
            mic=mic,
            mic_mask=mic_mask,
            mic_threshold=thresholds,
            causal=causal,
            iee_zero_tolerance=IEE_ZERO_TOLERANCE,
            scada_values_sha256=sample_digest(samples),
        )
    print(f"Wrote {len(starts)} MIC–IEE networks to {args.output}.")


if __name__ == "__main__":
    main()
