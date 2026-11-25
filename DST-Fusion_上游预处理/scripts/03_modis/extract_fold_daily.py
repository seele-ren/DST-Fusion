"""Extract county statistics and pixel support from daily MODIS index GeoTIFFs."""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from modis_indices import INDICATORS, STAT_SUFFIXES, get_valid_indices_and_ids, p10, p90


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--county", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_date > args.end_date or args.end_date.year > args.fold:
        parser.error("the date range must be ordered and end no later than the target year")

    with rasterio.open(args.mask) as src:
        tagged_fold = src.tags().get("target_year")
    if tagged_fold is not None and int(tagged_fold) != args.fold:
        parser.error("--fold does not match the mask's target_year tag")

    dates = [
        args.start_date + timedelta(days=offset)
        for offset in range((args.end_date - args.start_date).days + 1)
    ]
    prefix = "Jilin"
    paths = {
        day: {
            index: args.indices / f"{prefix}_{day.strftime('A%Y%j')}_{index}.tif"
            for index in INDICATORS
        }
        for day in dates
    }
    template = next((path for files in paths.values() for path in files.values() if path.exists()), None)
    if template is None:
        raise FileNotFoundError("No matching index GeoTIFFs in the requested date range.")
    with rasterio.open(template) as src:
        grid = src.shape, src.transform, src.crs
    indices, county_ids, names, codes = get_valid_indices_and_ids(
        str(template), str(args.county), str(args.mask), "name", "adcode"
    )
    base = pd.DataFrame({"县名": pd.Series(names), "区县代码": pd.Series(codes)})
    base.index.name = "id"
    base["mask_candidate_count"] = np.bincount(county_ids, minlength=len(names) + 1)[base.index]

    frames = []
    for day, day_paths in paths.items():
        rows = base.copy()
        rows.insert(0, "日期", day.isoformat())
        rows.insert(2, "fold", args.fold)
        for index, path in day_paths.items():
            columns = [f"{index}_{stat}" for stat in STAT_SUFFIXES]
            count = f"{index}_valid_count"
            if path.exists():
                with rasterio.open(path) as src:
                    if src.count != 1 or (src.shape, src.transform, src.crs) != grid:
                        raise ValueError(f"Index grid differs from template: {path}")
                    data = src.read(1, masked=True).ravel()[indices].astype("float64").filled(np.nan)
                valid = np.isfinite(data)
                values = pd.DataFrame({"id": county_ids[valid], "val": data[valid]})
                stats = values.groupby("id")["val"].agg(["median", "std", p10, p90, "size"])
                stats.columns = columns + [count]
                rows = rows.join(stats)
                rows[count] = rows[count].fillna(0).astype(int)
            else:
                rows[columns] = np.nan
                rows[count] = 0
            denominator = rows["mask_candidate_count"].replace(0, np.nan)
            rows[f"{index}_valid_fraction"] = rows[count] / denominator
        frames.append(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
