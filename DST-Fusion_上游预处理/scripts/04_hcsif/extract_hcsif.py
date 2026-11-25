"""Configure the archived HCSIF county extraction for annual or fixed fold masks."""

import argparse
from pathlib import Path

import rasterio

import tif_to_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sif-root", type=Path, required=True)
    parser.add_argument("--county", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mask = parser.add_mutually_exclusive_group(required=True)
    mask.add_argument("--ccd-root", type=Path, help="Use the observation-year CCD-Maize raster.")
    mask.add_argument("--fold-mask", type=Path, help="Use this fixed mask for every observation year.")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.end_year < args.start_year or args.workers < 1:
        parser.error("check the year range and number of workers")
    if args.fold_mask:
        with rasterio.open(args.fold_mask) as src:
            fold = src.tags().get("target_year")
        if fold is None or args.end_year > int(fold):
            parser.error("the fixed mask needs a target_year tag covering the requested years")
        tif_to_sequence.find_corn_file = lambda _year: str(args.fold_mask)
    else:
        tif_to_sequence.CCD_MAIZE_ROOT = str(args.ccd_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in range(args.start_year, args.end_year + 1):
        if (args.output_dir / f"Jilin_{year}_SIF_Stats_Full.csv").exists():
            parser.error(f"{year} output already exists; use a fresh output directory")
    tif_to_sequence.SIF_ROOT = str(args.sif_root)
    tif_to_sequence.COUNTY_SHP_PATH = str(args.county)
    tif_to_sequence.OUTPUT_DIR = str(args.output_dir)
    tif_to_sequence.START_YEAR = args.start_year
    tif_to_sequence.END_YEAR = args.end_year
    tif_to_sequence.MAX_WORKERS = args.workers
    tif_to_sequence.main()
    for year in range(args.start_year, args.end_year + 1):
        if not (args.output_dir / f"Jilin_{year}_SIF_Stats_Full.csv").is_file():
            raise RuntimeError(f"County extraction did not produce an output for {year}.")


if __name__ == "__main__":
    main()
