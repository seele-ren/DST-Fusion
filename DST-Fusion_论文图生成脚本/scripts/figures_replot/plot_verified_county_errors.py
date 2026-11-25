"""Render spatial bias from verified prediction files; never read plot labels from another run."""
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from pyproj import Geod
import shapely

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GEOJSON = ROOT / "code/data/吉林省.json"
SPECS = [
    ("DST-Fusion", "server rerun", ROOT / "runs/20261102_1543_box/rollval_predictions.csv",
     ROOT / "runs/20261102_1543_box/summary.json"),
    ("RF", "archived", ROOT / "code/outputs/rf_baseline/rollval_predictions.csv",
     ROOT / "code/outputs/rf_baseline/results.csv"),
    ("LSTM-daily", "archived", ROOT / "code/outputs/lstm_daily_baseline/rollval_predictions.csv",
     ROOT / "code/outputs/lstm_daily_baseline/results.csv"),
    ("Transformer-daily", "archived", ROOT / "code/outputs/transformer_daily_baseline/rollval_predictions.csv",
     ROOT / "code/outputs/transformer_daily_baseline/results.csv"),
    ("DDCN", "server rerun", ROOT / "runs/20261102_ddcn_map/rollval_predictions.csv",
     ROOT / "runs/20261102_ddcn_map/summary.json"),
    ("STGCN", "archived", ROOT / "code/outputs/stgcn_baseline/rollval_predictions.csv",
     ROOT / "code/outputs/stgcn_baseline/results.csv"),
]


def annual_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["val_year"])].append(row)
    results = []
    for year, group in sorted(groups.items()):
        yt = [float(r["y_true"]) for r in group]
        mean = statistics.mean(yt)
        sse = sum((float(r["y_pred"]) - float(r["y_true"]))**2 for r in group)
        sst = sum((y - mean)**2 for y in yt)
        results.append({"year": year, "n": len(group),
                        "rmse": math.sqrt(sse / len(group)), "r2": 1 - sse / sst})
    return results


def polygonal(geometry):
    valid = shapely.make_valid(geometry)
    if valid.geom_type == "GeometryCollection":
        parts = [g for g in valid.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        valid = shapely.union_all(parts)
    if valid.is_empty or valid.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Boundary repair lost the county's areal geometry")
    return valid


def scale_endpoint(geod, lon, lat, metres):
    low, high = lon, lon + 5
    for _ in range(45):
        middle = (low + high) / 2
        distance = geod.inv(lon, lat, middle, lat)[2]
        if distance < metres:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def main():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.55, "axes.unicode_minus": True,
                         "hatch.linewidth": 0.35})
    boundary = json.loads(GEOJSON.read_text())
    geo = gpd.GeoDataFrame.from_features(boundary["features"], crs="EPSG:4326")
    repaired = geo.loc[~geo.is_valid, "name"].tolist()
    geo["geometry"] = geo.geometry.map(polygonal)
    assert len(geo) == 60 and geo.is_valid.all() and not geo.geometry.is_empty.any()
    geo["city"] = [int(p["adcode"]) for p in geo["parent"]]
    cities = geo.dissolve(by="city")
    minx, miny, maxx, maxy = geo.total_bounds
    extent = [minx - 0.3, miny - 0.35, maxx + 0.3, maxy + 0.65]
    norm = mpl.colors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    cmap = mpl.colormaps["RdBu_r"]
    fig = plt.figure(figsize=(135 / 25.4, 187 / 25.4), layout="constrained")
    grid = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 0.075, 0.32])
    fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.025, wspace=0.025, hspace=0.045)
    per_county, manifest, model_rows = [], [], []
    canonical_keys = None
    reference = None
    geod = Geod(ellps="WGS84")
    for panel, (model, source_kind, path, check_path) in enumerate(SPECS):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        keys = {(int(r["val_year"]), r["county"]) for r in rows}
        assert len(rows) == 155 and len(keys) == 155
        assert {year for year, _ in keys} == set(range(2015, 2020))
        if canonical_keys is None:
            canonical_keys = keys
            reference = {(int(r["val_year"]), r["county"]): float(r["y_true"]) for r in rows}
        assert keys == canonical_keys
        for row in rows:
            key = (int(row["val_year"]), row["county"])
            assert abs(float(row["y_true"]) - reference[key]) < 1e-4
            assert abs(float(row["error"]) - (float(row["y_pred"]) - float(row["y_true"]))) < 1e-8
        annual = annual_metrics(rows)
        rmse = statistics.mean(r["rmse"] for r in annual)
        r2 = statistics.mean(r["r2"] for r in annual)
        if check_path.suffix == ".json":
            saved = json.loads(check_path.read_text())
            assert saved["all_checkpoint_predictions_and_metrics_verified"]
            expected_rmse, expected_r2 = saved["five_year_mean_rmse"], saved["five_year_mean_r2"]
        else:
            saved = list(csv.DictReader(check_path.open(encoding="utf-8-sig")))
            expected_rmse = statistics.mean(float(r["best_val_rmse"]) for r in saved)
            expected_r2 = statistics.mean(float(r["best_val_r2"]) for r in saved)
        assert abs(rmse - expected_rmse) < 1e-5 and abs(r2 - expected_r2) < 1e-5
        county = defaultdict(list)
        for row in rows:
            county[row["county"]].append(float(row["y_pred"]) - float(row["y_true"]))
        assert len(county) == 33 and set(county) <= set(geo["name"])
        bias = {name: statistics.mean(values) for name, values in county.items()}
        assert max(abs(v) for v in bias.values()) <= 4
        for name, values in county.items():
            per_county.append({"model": model, "county": name, "n_years": len(values),
                               "mean_signed_error": statistics.mean(values),
                               "mae": statistics.mean(abs(x) for x in values),
                               "rmse": math.sqrt(statistics.mean(x*x for x in values)),
                               "prediction_source": str(path.relative_to(ROOT))})
        frame = geo.copy()
        frame["bias"] = frame["name"].map(bias)
        ax = fig.add_subplot(grid[panel // 2, panel % 2])
        frame.plot(column="bias", ax=ax, cmap=cmap, norm=norm, linewidth=0.25,
                   edgecolor="#777777", missing_kwds={"color": "#D9D9D9", "hatch": "///",
                                                       "edgecolor": "#A6A6A6", "linewidth": 0.22})
        cities.boundary.plot(ax=ax, color="#404040", linewidth=0.65)
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
        ax.set_aspect(1 / math.cos(math.radians((miny + maxy) / 2)))
        ax.set_title(f"({chr(97 + panel)}) {model}" + ("*" if source_kind == "server rerun" else ""),
                     fontsize=9, fontweight="bold", pad=3)
        ax.text(0.985, 0.98, f"RMSE = {rmse:.3f}\n$R^2$ = {r2:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.6, linespacing=1.2,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
        ax.annotate("", xy=(0.43, 0.91), xytext=(0.43, 0.81), xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 0.7})
        ax.text(0.43, 0.965, "N", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        lon0, lat0 = minx + 0.65, miny + 0.15
        points = [lon0, scale_endpoint(geod, lon0, lat0, 100000),
                  scale_endpoint(geod, lon0, lat0, 200000)]
        ax.plot([points[0], points[-1]], [lat0, lat0], color="black", lw=0.7)
        for lon, label in zip(points, ["0", "", "200 km"]):
            ax.plot([lon, lon], [lat0, lat0 + 0.085], color="black", lw=0.65)
            ax.text(lon, lat0 + 0.12, label, fontsize=6.1, ha="center", va="bottom")
        ax.set_xticks([122, 126, 130], ["122°E", "126°E", "130°E"])
        ax.set_yticks([42, 44, 46], ["42°N", "44°N", "46°N"])
        ax.tick_params(labelsize=6.7, length=2.0, pad=1.5, width=0.5)
        item = {"model": model, "source_kind": source_kind,
                "prediction_file": str(path.relative_to(ROOT)),
                "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "crosscheck_file": str(check_path.relative_to(ROOT)),
                "mean_annual_rmse": rmse, "mean_annual_r2": r2,
                "annual_metrics": annual, "n_predictions": len(rows), "n_counties": len(county),
                "county_bias_range": [min(bias.values()), max(bias.values())]}
        manifest.append(item)
        model_rows.append({k: item[k] for k in ["model", "source_kind", "mean_annual_rmse", "mean_annual_r2"]})
    cax = fig.add_subplot(grid[3, :])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    bar.set_ticks([-4, -2, 0, 2, 4]); bar.ax.tick_params(labelsize=7, length=2, pad=1)
    bar.set_label("Mean signed error (Mg ha$^{-1}$)", fontsize=8, labelpad=2)
    bar.outline.set_linewidth(0.6)
    footer = fig.add_subplot(grid[4, :]); footer.set_axis_off()
    footer.legend(handles=[Patch(facecolor="#D9D9D9", edgecolor="#A6A6A6", hatch="///", label="Not evaluated"),
                           Line2D([], [], color="#777777", lw=0.4, label="County boundary"),
                           Line2D([], [], color="#404040", lw=0.8, label="City boundary")],
                  loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                  frameon=False, fontsize=6.8, handlelength=1.5, columnspacing=1.0)
    footer.text(0.5, 0.02, "* Server reruns; other panels: archived predictions.\n"
                "Panel metrics: five-year mean annual RMSE and $R^2$.",
                ha="center", va="bottom", transform=footer.transAxes, fontsize=6.8, linespacing=1.15)
    stem = HERE / "county_errors_server_predictions"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", metadata={"Title": "County mean signed yield errors, 2015-2019"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    with Image.open(stem.with_suffix(".png")) as image:
        rgb = image.convert("RGB")
        rgb.save(stem.with_suffix(".png"), dpi=(600, 600))
        rgb.resize((1080, round(rgb.height * 1080 / rgb.width))).save(HERE / "preview.png")
    plt.close(fig)
    for filename, values in [("county_error_values.csv", per_county), ("model_metrics.csv", model_rows)]:
        with (HERE / filename).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    provenance = {"models": manifest, "boundary_file": str(GEOJSON.relative_to(ROOT)),
                  "boundary_sha256": hashlib.sha256(GEOJSON.read_bytes()).hexdigest(),
                  "boundary_count": len(geo), "geometry_repairs": repaired,
                  "coordinate_assumption": "Supplied GeoJSON longitude/latitude treated as EPSG:4326; authoritative datum/version not documented in source.",
                  "aggregation": "Arithmetic mean of signed prediction-minus-observation errors over available years per county; 23 counties have five years, 10 have four.",
                  "missing": "27 county-level polygons have no evaluated prediction and are shown in gray hatching, never as zero.",
                  "color_limits": [-4, 4], "colormap": "RdBu_r", "out_of_range_values": 0,
                  "dimensions_mm": [135, 187], "png_dpi": 600,
                  "software": {"matplotlib": mpl.__version__, "geopandas": gpd.__version__, "shapely": shapely.__version__},
                  "original_manuscript_and_summary_tables_modified": False}
    (HERE / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(model_rows, indent=2))


if __name__ == "__main__":
    main()
