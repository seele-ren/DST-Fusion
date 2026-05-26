import argparse
import ast
import csv
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


@dataclass
class ModelSpec:
    title: str
    pred_csv: str
    results_csv: str


def _read_predictions(path: str, val_year: Optional[int] = None) -> Dict[str, List[float]]:
    by_county: Dict[str, List[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if val_year is not None:
                try:
                    row_year = int(float(row.get("val_year", "")))
                except (TypeError, ValueError):
                    continue
                if row_year != val_year:
                    continue
            county = (row.get("county") or "").strip()
            if not county:
                continue
            try:
                err = float(row["error"])
            except (KeyError, TypeError, ValueError):
                continue
            by_county.setdefault(county, []).append(err)
    return by_county


def _aggregate_error(by_county: Dict[str, List[float]], method: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for county, vals in by_county.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        if method == "median":
            out[county] = float(np.median(arr))
        else:
            out[county] = float(np.mean(arr))
    return out


def _read_metrics(results_csv: str, val_year: Optional[int] = None) -> Tuple[float, float]:
    rmse_vals: List[float] = []
    r2_vals: List[float] = []
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if val_year is not None:
                try:
                    row_year = int(float(row.get("val_year", "")))
                except (TypeError, ValueError):
                    continue
                if row_year != val_year:
                    continue
            try:
                rmse_vals.append(float(row["best_val_rmse"]))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                r2_vals.append(float(row["best_val_r2"]))
            except (KeyError, TypeError, ValueError):
                pass
    rmse = float(np.mean(rmse_vals)) if rmse_vals else float("nan")
    r2 = float(np.mean(r2_vals)) if r2_vals else float("nan")
    return rmse, r2


def _read_available_years(path: str) -> List[int]:
    years = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                years.add(int(float(row.get("val_year", ""))))
            except (TypeError, ValueError):
                continue
    return sorted(years)


def _default_geojson() -> str:
    candidates = glob.glob("data/*.json") or glob.glob("*.json")
    if not candidates:
        raise FileNotFoundError("No GeoJSON file found in current directory.")
    return sorted(candidates)[0]


def _build_specs(base_out: str) -> List[ModelSpec]:
    return [
        ModelSpec(
            title="DST-Fusion",
            pred_csv=os.path.join(base_out, "dual_tower", "rollval_predictions.csv"),
            results_csv=os.path.join(base_out, "dual_tower", "results.csv"),
        ),
        ModelSpec(
            title="RF",
            pred_csv=os.path.join(base_out, "rf_baseline", "rollval_predictions.csv"),
            results_csv=os.path.join(base_out, "rf_baseline", "results.csv"),
        ),
        ModelSpec(
            title="LSTM-daily",
            pred_csv=os.path.join(base_out, "lstm_daily_baseline", "rollval_predictions.csv"),
            results_csv=os.path.join(base_out, "lstm_daily_baseline", "results.csv"),
        ),
        ModelSpec(
            title="Transformer-daily",
            pred_csv=os.path.join(base_out, "transformer_daily_baseline", "rollval_predictions.csv"),
            results_csv=os.path.join(base_out, "transformer_daily_baseline", "results.csv"),
        ),
        ModelSpec(
            title="DDCN",
            pred_csv=os.path.join(base_out, "single_tower", "rollval_predictions.csv"),
            results_csv=os.path.join(base_out, "single_tower", "results.csv"),
        ),
        ModelSpec(
            title="STGCN",
            pred_csv=os.path.join(base_out, "stgcn_baseline", "rollval_predictions.csv"),
            results_csv=os.path.join(base_out, "stgcn_baseline", "results.csv"),
        ),
    ]


def _extract_city_id(parent_val, adcode_val):
    if isinstance(parent_val, dict) and "adcode" in parent_val:
        try:
            return int(parent_val["adcode"])
        except (TypeError, ValueError):
            pass
    if isinstance(parent_val, str) and parent_val.strip():
        txt = parent_val.strip()
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, dict) and "adcode" in parsed:
                return int(parsed["adcode"])
        except (ValueError, SyntaxError):
            pass
        nums = re.findall(r"\d{6}", txt)
        if nums:
            return int(nums[0])
    try:
        adcode = int(adcode_val)
        return (adcode // 100) * 100
    except (TypeError, ValueError):
        return None


def _save_all_formats(fig, out_path, dpi):
    stem, _ = os.path.splitext(out_path)
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".eps", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".tiff", dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {stem}.pdf")
    print(f"Wrote {stem}.eps")
    print(f"Wrote {stem}.tiff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot county-level choropleths for model errors.")
    parser.add_argument("--geojson", default=None, help="County GeoJSON path (default: first *.json in cwd)")
    parser.add_argument("--geo-county-col", default="name", help="County name column in GeoJSON")
    parser.add_argument("--out-dir", default="outputs/compare", help="Output directory")
    parser.add_argument("--out-name", default="county_error_choropleth_3x2", help="Output image file stem")
    parser.add_argument("--base-out", default="outputs", help="Base output directory containing model folders")
    parser.add_argument("--agg", choices=["mean", "median"], default="mean", help="Aggregation for county errors")
    parser.add_argument("--vmin", type=float, default=-4.0, help="Color scale min")
    parser.add_argument("--vmax", type=float, default=4.0, help="Color scale max")
    parser.add_argument("--dpi", type=int, default=1000, help="TIFF output DPI")
    parser.add_argument("--fig-width-mm", type=float, default=190.0, help="Final figure width in mm")
    parser.add_argument("--fig-height-mm", type=float, default=135.0, help="Final figure height in mm")
    parser.add_argument("--per-year", action="store_true", help="Generate one figure per validation year")
    parser.add_argument("--years", default="", help="Optional years to plot in per-year mode, e.g. 2015,2016")
    args = parser.parse_args()

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit("geopandas is required. Install with: pip install geopandas") from exc

    geojson = args.geojson or _default_geojson()
    specs = _build_specs(args.base_out)

    for spec in specs:
        if not os.path.exists(spec.pred_csv):
            raise FileNotFoundError(f"Missing prediction file: {spec.pred_csv}")
        if not os.path.exists(spec.results_csv):
            raise FileNotFoundError(f"Missing results file: {spec.results_csv}")

    gdf = gpd.read_file(geojson)
    if args.geo_county_col not in gdf.columns:
        raise KeyError(f"Geo county column not found: {args.geo_county_col}")
    if "adcode" not in gdf.columns and "parent" not in gdf.columns:
        raise KeyError("GeoJSON must contain 'parent' or 'adcode' columns to derive city boundaries.")

    gdf["_city_id"] = [
        _extract_city_id(row.get("parent"), row.get("adcode"))
        for _, row in gdf.iterrows()
    ]
    # Repair invalid geometries before dissolve to avoid GEOS TopologyException.
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    try:
        gdf = gdf[gdf.geometry.is_valid].copy()
    except Exception:
        pass
    city_boundary_gdf = None
    try:
        city_boundary_gdf = gdf.dropna(subset=["_city_id"]).dissolve(by="_city_id")
    except Exception as exc:
        print(f"Warning: failed to build city boundaries, skip overlay. {exc}")

    cmap = plt.get_cmap("RdBu_r")
    norm = mpl.colors.TwoSlopeNorm(vmin=args.vmin, vcenter=0.0, vmax=args.vmax)

    def draw_one_figure(year: Optional[int], out_name: str) -> None:
        n_models = len(specs)
        ncols = 3
        nrows = int(math.ceil(n_models / ncols))
        fig = plt.figure(
            figsize=(args.fig_width_mm / 25.4, args.fig_height_mm / 25.4),
            constrained_layout=False,
        )
        gs = fig.add_gridspec(nrows=nrows + 1, ncols=ncols, height_ratios=[1] * nrows + [0.08])
        map_axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]

        for i, spec in enumerate(specs):
            ax = map_axes[i]
            pred = _read_predictions(spec.pred_csv, val_year=year)
            err_map = _aggregate_error(pred, args.agg)
            rmse, r2 = _read_metrics(spec.results_csv, val_year=year)

            plot_gdf = gdf.copy()
            plot_gdf["err"] = plot_gdf[args.geo_county_col].map(err_map)
            plot_gdf.plot(
                column="err",
                ax=ax,
                cmap=cmap,
                norm=norm,
                linewidth=0.45,
                edgecolor="#8A8A8A",
                missing_kwds={
                    "color": "#f0f0f0",
                    "edgecolor": "#8A8A8A",
                    "linewidth": 0.45,
                },
            )
            if city_boundary_gdf is not None and len(city_boundary_gdf) > 0:
                city_boundary_gdf.boundary.plot(
                    ax=ax,
                    color="#444444",
                    linewidth=1.0,
                    zorder=5,
                )
            title = spec.title if year is None else f"{spec.title} ({year})"
            ax.set_title(title, fontsize=13.0, fontweight="bold", pad=5)
            ax.set_axis_off()
            ax.text(
                0.98,
                0.98,
                f"RMSE={rmse:.2f}\n$R^2$={r2:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 2.0},
            )

        for ax in map_axes[n_models:]:
            ax.set_axis_off()

        cax = fig.add_subplot(gs[nrows, :])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            fraction=0.08,
            pad=0.2,
            shrink=0.95,
        )
        cb.set_label("Prediction Error (Mg ha$^{-1}$)", fontsize=12.5, fontweight="bold")
        cb.set_ticks([args.vmin, -2.0, 0.0, 2.0, args.vmax])
        cax.tick_params(labelsize=10.5, width=1.0, direction="in")
        cb.outline.set_linewidth(1.0)
        fig.subplots_adjust(left=0.015, right=0.985, top=0.95, bottom=0.11, wspace=0.03, hspace=0.08)

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, out_name)
        _save_all_formats(fig, out_path, args.dpi)
        plt.close(fig)

    if args.per_year:
        if args.years.strip():
            years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        else:
            year_sets = [_read_available_years(spec.pred_csv) for spec in specs]
            years = sorted(set.intersection(*(set(s) for s in year_sets))) if year_sets else []
        if not years:
            raise SystemExit("No available years found for per-year plotting.")
        stem, ext = os.path.splitext(args.out_name)
        for year in years:
            draw_one_figure(year, f"{stem}_{year}")
    else:
        draw_one_figure(None, args.out_name)


if __name__ == "__main__":
    main()
