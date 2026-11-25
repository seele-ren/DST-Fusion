import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate and smooth county-level MODIS CSV time series."
    )
    parser.add_argument("--input-dir", default=r"E:\data\modis\Final_Results")
    parser.add_argument("--output-dir", default=r"E:\data\modis\Final_Results_Processed")
    parser.add_argument("--id-col", default="县名")
    parser.add_argument("--date-col", default="日期")
    parser.add_argument("--unit-col", default="区县代码")
    parser.add_argument("--window-length", type=int, default=15)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def normalize_window_length(window_length, series_length, polyorder):
    if series_length <= polyorder + 1:
        return None

    window = min(window_length, series_length)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return None
    return window


def smooth_series(series, window_length, polyorder):
    numeric = pd.to_numeric(series, errors="coerce")
    filled = numeric.interpolate(method="linear", limit_direction="both")

    if filled.isna().all():
        return filled

    window = normalize_window_length(window_length, len(filled), polyorder)
    if window is None:
        return filled

    try:
        return pd.Series(
            savgol_filter(filled.to_numpy(dtype=float), window_length=window, polyorder=polyorder),
            index=series.index,
        )
    except Exception:
        return filled


def process_one_county(group, target_cols, date_col, window_length, polyorder):
    group = group.sort_values(by=date_col).copy()
    for col in target_cols:
        group[col] = smooth_series(group[col], window_length, polyorder)
    return group


def make_group_keys(df, id_col, date_col, unit_col):
    if unit_col and unit_col in df.columns:
        return [unit_col]

    duplicate_count = df.duplicated([date_col, id_col], keep=False).sum()
    if duplicate_count == 0:
        return [id_col]

    df["_unit_seq"] = df.groupby([date_col, id_col]).cumcount()
    return [id_col, "_unit_seq"]


def process_file(input_path, output_path, id_col, date_col, unit_col, window_length, polyorder):
    df = read_csv(input_path)

    missing_cols = [col for col in [id_col, date_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    group_keys = make_group_keys(df, id_col, date_col, unit_col)
    protected_cols = [id_col, date_col] + group_keys + ["_unit_seq"]
    target_cols = [
        col for col in df.columns
        if col not in protected_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not target_cols:
        raise ValueError("no numeric indicator columns found")

    result = df.groupby(group_keys, group_keys=False).apply(
        lambda group: process_one_county(
            group, target_cols, date_col, window_length, polyorder
        )
    )
    result = result.drop(columns=["_unit_seq"], errors="ignore")

    result.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    return len(df), len(target_cols)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted(
        file_name for file_name in os.listdir(args.input_dir)
        if file_name.lower().endswith(".csv")
    )

    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Found {len(csv_files)} CSV files")

    processed = 0
    skipped = 0
    failed = 0

    for index, file_name in enumerate(csv_files, start=1):
        input_path = os.path.join(args.input_dir, file_name)
        output_path = os.path.join(args.output_dir, file_name)

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(csv_files)}] skip existing: {file_name}")
            continue

        try:
            rows, target_count = process_file(
                input_path,
                output_path,
                args.id_col,
                args.date_col,
                args.unit_col,
                args.window_length,
                args.polyorder,
            )
            processed += 1
            print(f"[{index}/{len(csv_files)}] ok: {file_name} rows={rows} columns={target_count}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(csv_files)}] fail: {file_name}: {exc}")

    print("=" * 40)
    print(f"processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
