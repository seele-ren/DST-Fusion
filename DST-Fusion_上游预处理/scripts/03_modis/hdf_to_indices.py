"""Run the archived MODIS index calculations with a pyhdf reader for two HDF4 tiles."""

import argparse
from contextlib import ExitStack
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from pyhdf.SD import SD, SDC

import batch_modis_pre
from modis_indices import INDICATORS


TILES = ("h26v04", "h27v04")
SIN = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
TILE_SIZE = 1111950.5196666666
GLOBAL_X, GLOBAL_Y = -20015109.354, 10007554.677003


def read_layer(hdfs, tile, band):
    sds = hdfs[tile].select(band)
    try:
        raw = sds[:]
        fill = sds.attributes()["_FillValue"]
    finally:
        sds.endaccess()

    size = 1200 if band == "state_1km_1" else 2400
    if raw.shape != (size, size):
        raise ValueError(f"{tile} {band}: unexpected shape {raw.shape}")
    values = raw.astype(np.float32)
    values[raw == fill] = np.nan
    resolution = TILE_SIZE / size
    x_origin = GLOBAL_X + int(tile[1:3]) * TILE_SIZE
    y_origin = GLOBAL_Y - int(tile[4:6]) * TILE_SIZE
    offsets = (np.arange(size) + 0.5) * resolution
    transform = Affine(resolution, 0, x_origin, 0, -resolution, y_origin)
    data = xr.DataArray(
        values, dims=("y", "x"), coords={"x": x_origin + offsets, "y": y_origin - offsets}, name=band,
    )
    return data.rio.write_crs(SIN).rio.write_transform(transform).rio.write_nodata(np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf-root", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    day = args.date.strftime("A%Y%j")
    paths = {}
    for tile in TILES:
        candidates = sorted(args.hdf_root.rglob(f"MOD09GA.{day}.{tile}.061.*.hdf"))
        if len(candidates) != 1:
            parser.error(f"{day} {tile}: expected one HDF file, found {len(candidates)}")
        paths[tile] = candidates[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = [args.output_dir / f"Jilin_{day}_{index}.tif" for index in INDICATORS]
    if any(path.exists() for path in expected):
        parser.error("index outputs already exist for this date; choose a fresh directory")

    with ExitStack() as stack:
        hdfs = {}
        for tile, path in paths.items():
            hdfs[tile] = SD(str(path), SDC.READ)
            stack.callback(hdfs[tile].end)
        batch_modis_pre.get_modis_layer = lambda tile, band: read_layer(hdfs, tile, band)
        message = batch_modis_pre.process_single_date(day, *TILES, args.output_dir)
    if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        raise RuntimeError(message)
    print(message)


if __name__ == "__main__":
    main()
