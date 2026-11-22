import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SAMPLE_INTERVAL = "10min"


def sample_digest(values):
    array = np.asarray(values, dtype="<f8", order="C")
    return sha256(array.tobytes()).hexdigest()


def read_scada(path, timestamp_column=None, variables=None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, float_precision="round_trip")
    elif path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path)
    else:
        raise ValueError("Use a CSV or XLSX input file.")
    if frame.empty:
        raise ValueError("The input has no SCADA records.")

    timestamp_column = timestamp_column or frame.columns[0]
    if variables is None:
        variables = frame.columns.drop(timestamp_column).tolist()
    if not variables or len(set(variables)) != len(variables) or timestamp_column in variables:
        raise ValueError("Select one or more distinct monitoring columns.")

    timestamps = pd.DatetimeIndex(pd.to_datetime(frame[timestamp_column], utc=True)).as_unit("us")
    if timestamps.hasnans or not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be present and strictly increasing.")
    values = frame[variables].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("An observed record contains a missing or non-finite measurement.")
    values.index = timestamps.rename("Timestamp")
    return values


def fill_missing_records(values):
    grid = pd.date_range(values.index[0], values.index[-1], freq=SAMPLE_INTERVAL, name="Timestamp")
    grid = grid.as_unit("us")
    if not values.index.isin(grid).all():
        raise ValueError("Observed timestamps do not lie on a common 10-minute grid.")
    filled = values.reindex(grid).interpolate(method="time", limit_area="inside")
    if not np.isfinite(filled.to_numpy()).all():
        raise ValueError("Missing measurements remain after interpolation.")
    return filled


def select_segment(values, start, records):
    start = pd.to_datetime(start, utc=True)
    if start not in values.index:
        raise ValueError(f"The requested start timestamp is unavailable: {start}.")
    position = values.index.get_loc(start)
    segment = values.iloc[position:position + records]
    if records <= 0 or len(segment) != records:
        raise ValueError("The source does not contain the requested segment length.")
    return segment


def standardize(values, reference=None):
    reference = values if reference is None else reference
    scaler = StandardScaler().fit(reference[values.columns])
    return pd.DataFrame(scaler.transform(values), index=values.index, columns=values.columns)


def main():
    parser = argparse.ArgumentParser(description="Prepare SCADA records using the preprocessing operations in the study.")
    parser.add_argument("operation", choices=["validate", "fill-gaps", "standardize"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="Write a new CSV file.")
    parser.add_argument("--time-column", help="Timestamp column (defaults to the first column).")
    parser.add_argument("--variables", nargs="+", help="Ordered measurement column names.")
    parser.add_argument("--start", help="Exact UTC timestamp at which to select a segment.")
    parser.add_argument("--records", type=int, help="Number of records to select with --start.")
    parser.add_argument("--reference", type=Path, help="SCADA file to fit Z-score parameters on.")
    parser.add_argument("--fit-rows", type=int, help="Use the first N rows of the reference for fitting.")
    args = parser.parse_args()

    if (args.start is None) != (args.records is None):
        parser.error("--start and --records must be supplied together.")
    if args.operation != "standardize" and (args.reference is not None or args.fit_rows is not None):
        parser.error("--reference and --fit-rows apply to standardize.")

    values = read_scada(args.input, args.time_column, args.variables)
    input_times = values.index
    if args.operation == "fill-gaps":
        values = fill_missing_records(values)
    if args.start is not None:
        values = select_segment(values, args.start, args.records)

    if args.operation == "standardize":
        reference = values
        if args.reference is not None:
            reference = read_scada(args.reference, args.time_column, values.columns.tolist())
        if args.fit_rows is not None:
            if not 0 < args.fit_rows <= len(reference):
                parser.error("--fit-rows must be within the number of available reference records.")
            reference = reference.iloc[:args.fit_rows]
        values = standardize(values, reference)

    output = values.reset_index()
    output["Timestamp"] = [timestamp.isoformat() for timestamp in values.index]
    output.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.17g", mode="x")
    if args.operation == "fill-gaps":
        print(f"Wrote {len(values)} records, including {(~values.index.isin(input_times)).sum()} interpolated records.")
    elif args.operation == "standardize":
        print(f"Wrote {len(values)} standardized records.")
    else:
        print(f"Wrote {len(values)} validated records.")


if __name__ == "__main__":
    main()
