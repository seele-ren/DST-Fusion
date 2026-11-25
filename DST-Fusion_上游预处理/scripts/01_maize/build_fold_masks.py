"""Build pre-target-year maize occurrence masks on the native CCD-Maize grid."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.windows import Window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccd-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--folds", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019])
    parser.add_argument("--percent", type=int, default=30)
    args = parser.parse_args()

    folds = sorted(set(args.folds))
    years = range(args.start_year, folds[-1])
    if folds[0] <= args.start_year or len(years) > 65535:
        parser.error("each fold needs 1 to 65,535 prior years")
    if not 1 <= args.percent <= 100:
        parser.error("--percent must be between 1 and 100")

    paths = {
        year: args.ccd_root / str(year) / f"classified-Jilin-maize-{year}-WGS84-v1.tif"
        for year in years
    }
    thresholds = {fold: (args.percent * (fold - args.start_year) + 99) // 100 for fold in folds}
    names = {fold: f"Jilin_fold_{fold}_freq{args.percent:02d}_30m.tif" for fold in folds}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in names.values()):
        parser.error("an output already exists; use a fresh output directory")

    with TemporaryDirectory(dir=args.output_dir) as staging:
        with rasterio.Env(GDAL_CACHEMAX=128 * 1024 * 1024), ExitStack() as stack:
            sources = {year: stack.enter_context(rasterio.open(path)) for year, path in paths.items()}
            first = sources[args.start_year]
            grid = first.shape, first.transform, first.crs
            if first.crs is None or first.crs.to_epsg() != 4326:
                raise ValueError("Expected the WGS84-v1 CCD-Maize annual maps.")
            for year, source in sources.items():
                if (source.shape, source.transform, source.crs) != grid or source.count != 1:
                    raise ValueError(f"{year}: expected one band on the common CCD-Maize grid.")

            profile = dict(
                driver="GTiff", width=first.width, height=first.height, count=1, dtype="uint8",
                crs=first.crs, transform=first.transform, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate", zlevel=3, nbits=1, BIGTIFF="IF_SAFER",
            )
            outputs = {
                fold: stack.enter_context(rasterio.open(Path(staging) / names[fold], "w", **profile))
                for fold in folds
            }
            totals = dict.fromkeys(folds, 0)
            for fold, output in outputs.items():
                output.update_tags(
                    target_year=fold, first_source_year=args.start_year, last_source_year=fold - 1,
                    minimum_occurrences=thresholds[fold], frequency_percent=args.percent,
                )

            for top in range(0, first.height, 256):
                height = min(256, first.height - top)
                window = Window(0, top, first.width, height)
                count = np.zeros((height, first.width), dtype=np.uint16)
                for year in range(args.start_year, folds[-1] + 1):
                    if year in outputs:
                        mask = count >= thresholds[year]
                        outputs[year].write(mask.astype("uint8"), 1, window=window)
                        totals[year] += int(np.count_nonzero(mask))
                    if year in sources:
                        source = sources[year]
                        pixels = source.read(1, window=window)
                        if (
                            MaskFlags.all_valid not in source.mask_flag_enums[0]
                            and not source.read_masks(1, window=window).all()
                        ):
                            raise ValueError(f"{year}: missing pixels in rows {top}–{top + height - 1}.")
                        if not ((pixels == 0) | (pixels == 1)).all():
                            raise ValueError(f"{year}: expected class codes 0 and 1.")
                        count += pixels.astype("uint16", copy=False)

        for fold, name in names.items():
            (Path(staging) / name).rename(args.output_dir / name)
            print(f"{name}: {totals[fold]:,} pixels, minimum {thresholds[fold]} occurrences", flush=True)


if __name__ == "__main__":
    main()
