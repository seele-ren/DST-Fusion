import argparse
import csv
import math
import os
from datetime import datetime, timedelta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 8-day SIF stats to weekly stats using overlap-weighted averages."
    )
    parser.add_argument(
        "--input-dir",
        default="Final_Results",
        help="Directory with yearly *_SIF_Stats_Full.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="Final_Results_weekly",
        help="Directory to write weekly CSVs.",
    )
    parser.add_argument(
        "--week-start-doy",
        type=int,
        default=128,
        help="Start DOY for week 0.",
    )
    parser.add_argument(
        "--week-end-doy",
        type=int,
        default=302,
        help="End DOY of the weekly coverage window (inclusive).",
    )
    parser.add_argument(
        "--week-length",
        type=int,
        default=7,
        help="Week length in days.",
    )
    parser.add_argument(
        "--sif-length",
        type=int,
        default=8,
        help="SIF composite length in days.",
    )
    parser.add_argument(
        "--min-overlap-days",
        type=float,
        default=3.0,
        help="If total overlap days < this, treat as missing.",
    )
    return parser.parse_args()


def overlap_length(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_week_intervals(start_doy, end_doy, week_length):
    last_start = end_doy - week_length + 1
    if last_start < start_doy:
        return []
    starts = list(range(start_doy, last_start + 1, week_length))
    return [(s, s + week_length) for s in starts]


def doy_to_date(year, doy):
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def _safe_float(val):
    if val is None:
        return math.nan
    if isinstance(val, float):
        return val
    text = str(val).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def to_weekly(rows, week_intervals, sif_cols, sif_length, min_overlap_days):
    output_rows = []
    by_county_year = {}

    for row in rows:
        date_int = int(row["日期"])
        year = date_int // 1000
        doy = date_int % 1000
        county = row["县名"]

        sif_values = [_safe_float(row.get(col, "")) for col in sif_cols]
        key = (county, year)
        by_county_year.setdefault(key, {"year": year, "entries": []})
        by_county_year[key]["entries"].append((doy, sif_values))

    for (county, _), info in by_county_year.items():
        entries = sorted(info["entries"], key=lambda x: x[0])
        year = int(info["year"])
        sif_start = [e[0] for e in entries]
        sif_end = [d + sif_length for d in sif_start]
        sif_values = [e[1] for e in entries]

        for week_index, (w_start, w_end) in enumerate(week_intervals):
            weights = [
                overlap_length(w_start, w_end, s_start, s_end)
                for s_start, s_end in zip(sif_start, sif_end)
            ]
            total_overlap = float(sum(weights))
            weekly_vals = []
            missing_flag = 0
            for col_idx in range(len(sif_cols)):
                weighted_sum = 0.0
                overlap_valid = 0.0
                for row_idx, w in enumerate(weights):
                    val = sif_values[row_idx][col_idx]
                    if w <= 0.0 or math.isnan(val):
                        continue
                    weighted_sum += w * val
                    overlap_valid += w
                if overlap_valid < min_overlap_days:
                    weekly_vals.append(math.nan)
                    missing_flag = 1
                else:
                    weekly_vals.append(weighted_sum / overlap_valid)
            if total_overlap < min_overlap_days:
                missing_flag = 1

            start_date = doy_to_date(year, w_start)
            end_date = doy_to_date(year, w_end - 1)
            out_row = {
                "year": year,
                "week_index": week_index,
                "week_start_doy": w_start,
                "week_end_doy": w_end - 1,
                "week_start_date": start_date.strftime("%Y-%m-%d"),
                "week_end_date": end_date.strftime("%Y-%m-%d"),
                "县名": county,
                "sif_missing_flag": missing_flag,
                "sif_overlap_days": total_overlap,
            }
            out_row.update(dict(zip(sif_cols, weekly_vals)))
            output_rows.append(out_row)

    return output_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    week_intervals = build_week_intervals(
        args.week_start_doy, args.week_end_doy, args.week_length
    )

    files = [
        f
        for f in os.listdir(args.input_dir)
        if f.endswith("_SIF_Stats_Full.csv")
    ]
    if not files:
        raise FileNotFoundError(f"No *_SIF_Stats_Full.csv files in {args.input_dir}")

    for fname in sorted(files):
        in_path = os.path.join(args.input_dir, fname)
        with open(in_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        sif_cols = [c for c in rows[0].keys() if c.startswith("SIF_")]
        if not sif_cols:
            raise ValueError(f"No SIF columns found in {in_path}")

        weekly_rows = to_weekly(
            rows,
            week_intervals,
            sif_cols,
            args.sif_length,
            args.min_overlap_days,
        )

        out_name = fname.replace("_SIF_Stats_Full.csv", "_SIF_Weekly.csv")
        out_path = os.path.join(args.output_dir, out_name)
        fieldnames = [
            "year",
            "week_index",
            "week_start_doy",
            "week_end_doy",
            "week_start_date",
            "week_end_date",
            "县名",
            "sif_missing_flag",
            "sif_overlap_days",
        ] + sif_cols
        with open(out_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in weekly_rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
