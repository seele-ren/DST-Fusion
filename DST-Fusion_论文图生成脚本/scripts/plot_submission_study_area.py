"""Render Remote Sensing Figure 1 from the study boundaries and annual CCD-Maize maps."""

import argparse
import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from PIL import Image
from pyproj import Geod
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.plot import plotting_extent
from rasterio.transform import from_bounds

from plot_choropleth_models import _longitude_for_distance


YEARS = range(2001, 2020)
MAIZE_THRESHOLD = 0.30
DISPLAY_SCALE = 0.1
MAIZE_COLOR = "#417A59"
PROVINCE_FILL = "#F5F5F0"
PROVINCE_EDGE = "#20282D"
COUNTY_EDGE = "#344857"


def read_boundaries(boundary_file, prediction_file):
    features = json.loads(boundary_file.read_text(encoding="utf-8"))["features"]
    boundaries = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    boundaries.geometry = boundaries.make_valid(method="structure", keep_collapsed=False)
    if not boundaries.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Every county boundary must be a polygon or multipolygon.")
    with prediction_file.open(encoding="utf-8-sig", newline="") as stream:
        names = {row["county"] for row in csv.DictReader(stream)}
    counties = boundaries.loc[boundaries["name"].isin(names)]
    if len(names) != 33 or len(counties) != 33 or set(counties["name"]) != names:
        raise ValueError("Expected a one-to-one boundary match for all 33 modeling units.")
    return boundaries.geometry.union_all(), counties


def read_maize_layer(data_dir, province, save_mask=None):
    paths = [data_dir / str(year) / f"classified-Jilin-maize-{year}-WGS84-v1.tif" for year in YEARS]
    with rasterio.open(paths[0]) as source:
        if source.crs is None or source.crs.to_epsg() != 4326:
            raise ValueError("The annual maize maps must be in WGS84.")
        source_grid = (source.shape, source.transform, source.crs)
        height = int(source.height * DISPLAY_SCALE)
        width = int(source.width * DISPLAY_SCALE)
        transform = from_bounds(*source.bounds, width, height)

    counts = np.zeros((height, width), dtype=np.uint8)
    for path in paths:
        with rasterio.open(path) as source:
            if (source.shape, source.transform, source.crs) != source_grid:
                raise ValueError(f"The grid of {path.name} differs from the 2001 grid.")
            pixels = source.read(1, out_shape=counts.shape, resampling=Resampling.nearest)
        if np.any(pixels > 1):
            raise ValueError(f"{path.name} contains values outside the binary CCD-Maize classes.")
        counts += pixels

    inside_province = geometry_mask(
        [province], out_shape=counts.shape, transform=transform, invert=True
    )
    maize = inside_province & (counts >= math.ceil(MAIZE_THRESHOLD * len(paths)))
    print(f"Annual maps: {len(paths)}. Display grid: {width} by {height}. Maize pixels: {maize.sum()}.")
    if save_mask is not None:
        save_mask.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            save_mask, "w", driver="GTiff", width=width, height=height, count=1,
            dtype="uint8", crs="EPSG:4326", transform=transform, compress="deflate",
        ) as output:
            output.write(maize.astype("uint8"), 1)
            output.update_tags(
                source="CCD-Maize WGS84-v1", start_year=YEARS[0], end_year=YEARS[-1],
                minimum_occurrences=math.ceil(MAIZE_THRESHOLD * len(paths)), grid_scale=DISPLAY_SCALE,
            )
    return np.ma.masked_where(~maize, maize), plotting_extent(maize, transform)


def draw_map(province, counties, maize, extent, pdf_path, png_path):
    figure, axes = plt.subplots(figsize=(6.15, 4.85))
    figure.subplots_adjust(left=0.082, right=0.985, bottom=0.113, top=0.975)
    province_layer = gpd.GeoSeries([province], crs=counties.crs)
    province_layer.plot(ax=axes, color=PROVINCE_FILL, zorder=1)
    axes.imshow(
        maize, extent=extent, origin="upper", interpolation="nearest",
        cmap=ListedColormap([MAIZE_COLOR]), zorder=2,
    )
    counties.boundary.plot(ax=axes, color=COUNTY_EDGE, linewidth=0.60, zorder=3)
    province_layer.boundary.plot(ax=axes, color=PROVINCE_EDGE, linewidth=0.86, zorder=4)

    west, south, east, north = province.bounds
    axes.set_xlim(west - 0.34, east + 0.34)
    axes.set_ylim(south - 0.46, north + 0.18)
    axes.set_aspect(1 / math.cos(math.radians((south + north) / 2)))
    axes.xaxis.set_major_locator(MultipleLocator(2))
    axes.yaxis.set_major_locator(MultipleLocator(1))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}°E"))
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}°N"))
    axes.set_xlabel("")
    axes.set_ylabel("")
    axes.tick_params(labelsize=8.5, length=2.5, width=0.6, pad=3)
    axes.spines[["top", "right"]].set_visible(False)
    for spine in axes.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#778188")

    axes.annotate(
        "", xy=(0.948, 0.895), xytext=(0.948, 0.775), xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": PROVINCE_EDGE},
    )
    axes.text(
        0.948, 0.92, "N", transform=axes.transAxes, ha="center", va="bottom",
        fontsize=10, fontweight="bold", color=PROVINCE_EDGE,
    )
    longitude, latitude = west + 0.19, south - 0.21
    geod = Geod(ellps="WGS84")
    endpoint = _longitude_for_distance(geod, longitude, latitude, 100_000)
    axes.plot([longitude, endpoint], [latitude, latitude], color=PROVINCE_EDGE, linewidth=1.0)
    for x, label in [(longitude, "0"), (endpoint, "100 km")]:
        axes.plot([x, x], [latitude, latitude + 0.085], color=PROVINCE_EDGE, linewidth=0.8)
        axes.text(x, latitude + 0.13, label, ha="center", va="bottom", fontsize=8)

    legend_items = [
        Patch(facecolor=MAIZE_COLOR, label="Maize occurrence ≥30% (2001-2019)"),
        Line2D([], [], color=COUNTY_EDGE, lw=0.60, label="County-level boundary"),
        Line2D([], [], color=PROVINCE_EDGE, lw=0.86, label="Provincial boundary"),
    ]
    figure.legend(
        handles=legend_items, loc="lower center", bbox_to_anchor=(0.53, 0.013),
        ncols=3, frameon=False, fontsize=8.1, columnspacing=1.35, handlelength=1.9,
    )
    figure.savefig(pdf_path, dpi=600, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    with Image.open(png_path) as source:
        rgb = source.convert("RGB")
    rgb.save(png_path, dpi=(600, 600))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ccd-maize", type=Path)
    source.add_argument("--display-mask", type=Path)
    parser.add_argument("--save-display-mask", type=Path)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    province, counties = read_boundaries(args.boundaries, args.predictions)
    if args.display_mask:
        if args.save_display_mask:
            parser.error("--save-display-mask requires --ccd-maize")
        with rasterio.open(args.display_mask) as dataset:
            maize = dataset.read(1).astype(bool)
            extent = plotting_extent(maize, dataset.transform)
        maize = np.ma.masked_where(~maize, maize)
    else:
        maize, extent = read_maize_layer(args.ccd_maize, province, args.save_display_mask)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    draw_map(province, counties, maize, extent, args.pdf, args.png)


if __name__ == "__main__":
    main()
