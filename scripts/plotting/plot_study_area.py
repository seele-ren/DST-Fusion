import argparse
import glob
import os
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent
from rasterio.windows import from_bounds


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


COLOR_PROVINCE = "#4F81BD"
COLOR_COUNTY_EDGE = "#5A5A5A"
COLOR_CROP = "#3C8D40"
COLOR_INSET_BG = "#F4F4F4"


def _default_geojson() -> str:
    candidates = sorted(glob.glob("data/*.json") or glob.glob("*.json"))
    if not candidates:
        raise FileNotFoundError("No GeoJSON file found in current directory.")
    return candidates[0]


def _save_all_formats(fig: plt.Figure, out_path: str, dpi: int) -> None:
    stem, _ = os.path.splitext(out_path)
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".eps", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".tiff", dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {stem}.pdf")
    print(f"Wrote {stem}.eps")
    print(f"Wrote {stem}.tiff")


def _read_raster(
    path: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    downsample: int = 1,
    resampling: Resampling = Resampling.bilinear,
):
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform) if bounds is not None else None
        if window is None:
            out_h = max(1, src.height // downsample)
            out_w = max(1, src.width // downsample)
            data = src.read(1, out_shape=(out_h, out_w), resampling=resampling)
            transform = src.transform * src.transform.scale(src.width / data.shape[1], src.height / data.shape[0])
        else:
            out_h = max(1, int(window.height) // downsample)
            out_w = max(1, int(window.width) // downsample)
            data = src.read(1, window=window, out_shape=(out_h, out_w), resampling=resampling)
            win_transform = src.window_transform(window)
            transform = win_transform * win_transform.scale(window.width / data.shape[1], window.height / data.shape[0])
        extent = plotting_extent(data, transform)
        return data, extent, src.crs


def _hillshade(dem: np.ndarray) -> np.ndarray:
    arr = dem.astype(np.float64)
    arr[~np.isfinite(arr)] = np.nan
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(arr, vert_exag=0.5)


def _infer_crop_mask(mask: np.ndarray) -> np.ndarray:
    vals = mask[np.isfinite(mask)]
    if vals.size == 0:
        return np.full_like(mask, np.nan, dtype=np.float32)
    unique_vals = np.unique(vals)
    if unique_vals.size <= 6 and 1 in unique_vals:
        return np.where(mask == 1, 1.0, np.nan).astype(np.float32)
    threshold = float(np.nanpercentile(vals, 75))
    return np.where(mask >= threshold, 1.0, np.nan).astype(np.float32)


def _add_panel_tag(ax, tag: str) -> None:
    ax.text(
        0.015,
        0.985,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.0},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot publication-ready study area figure (a, b, c).")
    parser.add_argument("--geojson", default=None, help="County/province GeoJSON path")
    parser.add_argument("--dem", default="", help="Optional DEM GeoTIFF")
    parser.add_argument("--crop-mask", default="", help="Optional crop mask GeoTIFF")
    parser.add_argument("--out", default="outputs/compare/study_area", help="Output file stem")
    parser.add_argument("--dpi", type=int, default=1000, help="TIFF export DPI")
    parser.add_argument("--fig-width-mm", type=float, default=190.0, help="Final figure width in mm")
    parser.add_argument("--fig-height-mm", type=float, default=135.0, help="Final figure height in mm")
    parser.add_argument("--pad-deg", type=float, default=0.25, help="Map extent padding in degrees")
    parser.add_argument("--downsample", type=int, default=4, help="Raster downsample factor")
    args = parser.parse_args()

    geojson = args.geojson or _default_geojson()
    gdf = gpd.read_file(geojson)
    if gdf.crs is None:
        raise ValueError("GeoJSON has no CRS.")
    gdf = gdf.to_crs(epsg=4326)
    province_shape = gdf.dissolve()

    minx, miny, maxx, maxy = gdf.total_bounds
    bounds = (
        minx - args.pad_deg,
        miny - args.pad_deg,
        maxx + args.pad_deg,
        maxy + args.pad_deg,
    )

    dem_arr = dem_extent = dem_img = None
    crop_arr = crop_extent = None
    if args.dem:
        dem_arr, dem_extent, _ = _read_raster(
            args.dem,
            bounds=bounds,
            downsample=max(1, args.downsample),
            resampling=Resampling.bilinear,
        )
        dem_arr = dem_arr.astype(np.float32)
        dem_arr[~np.isfinite(dem_arr)] = np.nan
    if args.crop_mask:
        crop_arr, crop_extent, _ = _read_raster(
            args.crop_mask,
            bounds=bounds,
            downsample=max(1, args.downsample),
            resampling=Resampling.nearest,
        )
        crop_arr = _infer_crop_mask(crop_arr)

    fig = plt.figure(figsize=(args.fig_width_mm / 25.4, args.fig_height_mm / 25.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax in [ax_a, ax_b, ax_c]:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#4A4A4A")

    _add_panel_tag(ax_a, "(a)")
    ax_a.set_facecolor(COLOR_INSET_BG)
    province_shape.plot(ax=ax_a, facecolor=COLOR_PROVINCE, edgecolor="#2F4F6F", linewidth=0.9)
    ax_a.set_title("Jilin Province", fontsize=13.0, fontweight="bold", pad=4)
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    _add_panel_tag(ax_b, "(b)")
    if dem_arr is not None and dem_extent is not None:
        hill = _hillshade(dem_arr)
        ax_b.imshow(hill, extent=dem_extent, origin="upper", cmap="Greys", alpha=0.55, interpolation="bilinear")
        dem_img = ax_b.imshow(
            dem_arr,
            extent=dem_extent,
            origin="upper",
            cmap="Greys",
            alpha=0.22,
            interpolation="bilinear",
        )
    gdf.plot(ax=ax_b, facecolor="none", edgecolor=COLOR_COUNTY_EDGE, linewidth=0.45)
    province_shape.boundary.plot(ax=ax_b, color="#1F1F1F", linewidth=1.0)
    ax_b.set_title("County Boundaries and Terrain", fontsize=13.0, fontweight="bold", pad=4)
    ax_b.set_xlim(bounds[0], bounds[2])
    ax_b.set_ylim(bounds[1], bounds[3])
    ax_b.tick_params(labelsize=10.0, width=1.0, direction="in")
    ax_b.set_xlabel("Longitude", fontsize=12.0, fontweight="bold")
    ax_b.set_ylabel("Latitude", fontsize=12.0, fontweight="bold")

    _add_panel_tag(ax_c, "(c)")
    if dem_arr is not None and dem_extent is not None:
        ax_c.imshow(_hillshade(dem_arr), extent=dem_extent, origin="upper", cmap="Greys", alpha=0.35, interpolation="bilinear")
    if crop_arr is not None and crop_extent is not None:
        ax_c.imshow(
            crop_arr,
            extent=crop_extent,
            origin="upper",
            cmap=ListedColormap([COLOR_CROP]),
            alpha=1.0,
            interpolation="nearest",
        )
    gdf.plot(ax=ax_c, facecolor="none", edgecolor=COLOR_COUNTY_EDGE, linewidth=0.4)
    province_shape.boundary.plot(ax=ax_c, color="#1F1F1F", linewidth=1.0)
    ax_c.set_title("Crop Distribution", fontsize=13.0, fontweight="bold", pad=4)
    ax_c.set_xlim(bounds[0], bounds[2])
    ax_c.set_ylim(bounds[1], bounds[3])
    ax_c.tick_params(labelsize=10.0, width=1.0, direction="in")
    ax_c.set_xlabel("Longitude", fontsize=12.0, fontweight="bold")
    ax_c.set_ylabel("Latitude", fontsize=12.0, fontweight="bold")
    ax_c.legend(
        handles=[Patch(facecolor=COLOR_CROP, edgecolor="none", label="Crop mask")],
        loc="lower left",
        frameon=False,
        fontsize=10.0,
    )

    if dem_img is not None:
        cbar = fig.colorbar(dem_img, ax=[ax_b, ax_c], orientation="vertical", fraction=0.025, pad=0.015)
        cbar.set_label("Elevation (m)", fontsize=12.0, fontweight="bold")
        cbar.ax.tick_params(labelsize=10.0, width=1.0, direction="in")
        cbar.outline.set_linewidth(1.0)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.08, wspace=0.16, hspace=0.18)
    _save_all_formats(fig, args.out, args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
