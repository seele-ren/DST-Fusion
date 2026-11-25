"""Join annual optical statistics and county meteorology by date and county."""

import argparse
from pathlib import Path

import pandas as pd


def read_optical(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veg-dir", type=Path, required=True)
    parser.add_argument("--climate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2019)
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("check the year range")

    stats = [
        f"{index}_{suffix}"
        for index in ("EVI", "WDRVI", "GCI", "NDWI", "NIRv")
        for suffix in ("median", "std", "p10", "p90")
    ]
    weather = ["temp_avg_c", "tmax_c", "GDD", "KDD", "PRCP", "VPD"]
    keys = ["Date", "County"]
    merged = []
    for year in range(args.start_year, args.end_year + 1):
        optical = read_optical(args.veg_dir / f"Jilin_{year}_Stats_Full.csv").rename(columns={"县名": "County"})
        climate = pd.read_csv(args.climate_dir / f"吉林_气候数据_{year}.csv", encoding="utf-8-sig")
        optical["Date"] = pd.to_datetime(optical["日期"], format="A%Y%j")
        climate["Date"] = pd.to_datetime(climate["time"])
        joined = optical[keys + stats].merge(climate[keys + weather], on=keys, validate="one_to_one")
        if joined.empty:
            raise ValueError(f"{year}: no matching optical and climate rows")
        merged.append(joined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(merged, ignore_index=True).to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
