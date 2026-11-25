"""Build the six-model spatial diagnostic from the verified Figure 4 inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from esda import Moran
from libpysal.weights import W
from pyproj import Geod
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FIGURE_SOURCE = PAPER / "figures_replot/provenance.json"
YEARS = dict(zip(range(2015, 2020), [33, 33, 28, 28, 33], strict=True))
ISLANDS = {"公主岭市", "伊通满族自治县"}
SEED = 42
PERMUTATIONS = 9_999
FIGURE_SIZE = (180 / 25.4, 190 / 25.4)
MISSING = {"color": "#D9DADB", "edgecolor": "#777C83", "hatch": "///", "linewidth": 0.20}
UNITS = r"(Mg ha$^{-1}$)"

sys.path.insert(0, str(FIGURE_SOURCE.parent))
from plot_verified_county_errors import polygonal, scale_endpoint


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions(project):
    source = json.loads(FIGURE_SOURCE.read_text())
    specs = source["models"]
    assert len(specs) == 6 and specs[3]["model"] == "Transformer-daily"
    tables = []
    for model in specs:
        path = project / model["prediction_file"]
        assert sha256(path) == model["prediction_sha256"], path
        frame = pd.read_csv(path).rename(columns={"val_year": "year"})
        frame = frame.set_index(["year", "county"]).sort_index()
        assert frame.index.is_unique
        assert frame.groupby(level="year").size().to_dict() == YEARS
        assert np.isfinite(frame[["y_true", "y_pred", "error"]]).all().all()
        error = frame["y_pred"] - frame["y_true"]
        np.testing.assert_allclose(error, frame["error"], rtol=0, atol=1e-8)
        if tables:
            pd.testing.assert_index_equal(frame.index, tables[0].index)
            np.testing.assert_allclose(frame["y_true"], tables[0]["observed"], rtol=0, atol=1e-6)
        else:
            assert frame.index.get_level_values("county").nunique() == 33
        tables.append(pd.DataFrame({"model": model["model"], "observed": frame["y_true"],
                                    "estimated": frame["y_pred"], "signed_error": error,
                                    "absolute_error": error.abs()}, index=frame.index))
    return source, pd.concat(tables).reset_index()


def moran_table(county, graph_path, model_order):
    names = sorted(county["county"].unique())
    neighbors = {name: [] for name in names}
    edges = pd.read_csv(graph_path)
    for edge in edges.itertuples(index=False):
        assert edge.county_a != edge.county_b and edge.county_b not in neighbors[edge.county_a]
        neighbors[edge.county_a].append(edge.county_b)
        neighbors[edge.county_b].append(edge.county_a)
    assert len(edges) == 38
    assert {name for name, adjacent in neighbors.items() if not adjacent} == ISLANDS
    retained = [name for name in names if name not in ISLANDS]
    weights = W({name: neighbors[name] for name in retained}, id_order=retained, silence_warnings=True)
    weights.transform = "r"
    matrix, _ = weights.full()
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    assert np.all(np.diag(matrix) == 0) and len(retained) == 31

    results = []
    for model in model_order:
        mae = county.loc[county["model"].eq(model)].set_index("county")["mae"]
        y = mae.reindex(retained).to_numpy()
        assert np.isfinite(y).all()
        np.random.seed(SEED)
        m = Moran(y, weights, transformation="r", permutations=PERMUTATIONS)
        z = y - y.mean()
        np.testing.assert_allclose(m.I, (z @ matrix @ z) / (z @ z), rtol=0, atol=1e-14)
        assert len(m.sim) == PERMUTATIONS
        # The cutoff is the upper tail specified by the positive-autocorrelation hypothesis.
        p = (1 + np.count_nonzero(m.sim >= m.I)) / (PERMUTATIONS + 1)
        results.append({"model": model, "n": len(retained), "moran_i": m.I,
                        "p_permutation_upper": p})
    result = pd.DataFrame(results)
    result["p_holm"] = multipletests(result["p_permutation_upper"], method="holm")[1]
    return result


def read_geography(path):
    boundary = json.loads(path.read_text())
    geo = gpd.GeoDataFrame.from_features(boundary["features"], crs="EPSG:4326")
    repaired = geo.loc[~geo.is_valid, "name"].tolist()
    geo["geometry"] = geo.geometry.map(polygonal)
    assert len(geo) == 60 and geo["name"].is_unique and geo.geometry.is_valid.all()
    geo["prefecture"] = [int(parent["adcode"]) for parent in geo["parent"]]
    return geo, repaired


def draw_map(ax, geo, outer, values, cmap, norm, panel, show_lon, show_lat):
    frame = geo.assign(value=geo["name"].map(values))
    frame.plot(column="value", ax=ax, cmap=cmap, norm=norm, linewidth=0.22,
               edgecolor="#444A51", missing_kwds=MISSING, aspect=None)
    outer.plot(ax=ax, color="#363C44", linewidth=0.42, aspect=None)
    ax.set(xlim=(121.25, 132.30), ylim=(40.2, 46.65), xticks=[123, 127, 131], yticks=[42, 44, 46])
    ax.set(xlabel="", ylabel="")
    ax.set_aspect(1 / np.cos(np.deg2rad(43.5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}°N"))
    ax.tick_params(length=2, width=0.45, pad=1, labelsize=6.75, labelbottom=show_lon,
                   labelleft=show_lat, bottom=show_lon, left=show_lat)
    for spine in ax.spines.values():
        spine.set_linewidth(0.50)
        spine.set_color("#6F747A")
    ax.set_title(panel, fontsize=9.2, weight="bold", pad=2)

    # Identical extents allow a scale and north arrow in the first map of each group.
    if show_lat and not show_lon:
        ax.annotate("N", xy=(131.70, 45.88), xytext=(131.70, 46.34),
                    ha="center", va="center", fontsize=6.4)
        ax.annotate("", xy=(131.70, 46.17), xytext=(131.70, 45.48),
                    arrowprops={"arrowstyle": "-|>", "linewidth": 0.65, "color": "#24282C",
                                "shrinkA": 0, "shrinkB": 0})
        lon, lat = 121.85, 40.74
        end = scale_endpoint(Geod(ellps="WGS84"), lon, lat, 200_000)
        ax.plot([lon, end], [lat, lat], color="#24282C", lw=0.72)
        ax.vlines([lon, end], lat - 0.08, lat + 0.08, color="#24282C", lw=0.62)
        ax.text((lon + end) / 2, lat + 0.12, "200 km", ha="center", va="bottom", fontsize=6.4)


def draw_page(geo, outer, table, model_order, year=None):
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    is_annual = year is not None
    limit = 5 if is_annual else 3
    fields = ("signed_error", "absolute_error") if is_annual else ("mean_signed_error", "mae")
    headings = ("Signed error", "Absolute error") if is_annual else ("Mean signed error", "Mean absolute error")
    title = f"{year}  |  {YEARS[year]} evaluated counties" if is_annual else "2015–2019  |  33 evaluated counties"
    figure.text(0.055, 0.984, title, fontsize=10, color="#303942", va="top", weight="bold")
    for group, (field, heading, cmap_name) in enumerate(zip(fields, headings, ("RdBu_r", "YlOrBr"), strict=True)):
        top = 0.932 - 0.454 * group
        figure.text(0.055, top, f"{'AB'[group]}  {heading}", fontsize=10.2, va="top", weight="bold")
        low = -limit if group == 0 else 0
        norm = mpl.colors.Normalize(low, limit)
        cmap = mpl.colormaps[cmap_name]
        assert table[field].between(low, limit).all(), f"Display would clip {field}"
        for panel, model in enumerate(model_order):
            row, col = divmod(panel, 3)
            ax = figure.add_axes([0.087 + 0.301 * col, top - 0.197 - 0.158 * row, 0.278, 0.135])
            values = table.loc[table["model"].eq(model)].set_index("county")[field]
            assert values.index.is_unique and len(values) == (YEARS[year] if is_annual else 33)
            draw_map(ax, geo, outer, values, cmap, norm, model, show_lon=row == 1, show_lat=col == 0)
        cax = figure.add_axes([0.260, top - 0.395, 0.545, 0.014])
        bar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                              orientation="horizontal", ticks=np.linspace(low, limit, 7 if group == 0 and not is_annual else 6))
        if group == 0 and is_annual:
            bar.set_ticks([-5, -4, -2, 0, 2, 4, 5])
        if group == 1 and not is_annual:
            bar.set_ticks([0, 1, 2, 3])
        bar.ax.tick_params(labelsize=7.1, length=2, pad=1.5, width=0.5)
        bar.outline.set_linewidth(0.5)
        figure.text(0.819, top - 0.389, UNITS, fontsize=7.3, va="center")
    figure.legend(handles=[Patch(facecolor=MISSING["color"], edgecolor=MISSING["edgecolor"],
                                hatch=MISSING["hatch"], label="No evaluated prediction", linewidth=0.5)],
                  loc="lower center", bbox_to_anchor=(0.5, 0.006), frameon=False, fontsize=7.4)
    return figure


def main(project, graph, output):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": True,
                         "hatch.linewidth": 0.35, "text.color": "#1E2329"})
    source, annual = load_predictions(project)
    order = [m["model"] for m in source["models"]]
    county = annual.groupby(["model", "county"], sort=False).agg(
        n_years=("year", "size"), mean_signed_error=("signed_error", "mean"), mae=("absolute_error", "mean")
    ).reset_index()
    assert len(annual) == 930 and len(county) == 198
    assert county.groupby("model")["n_years"].value_counts().unstack().eq({4: 10, 5: 23}).all().all()
    assert np.all(county["mean_signed_error"].abs() <= county["mae"] + 1e-12)
    for model in source["models"]:
        values = county.loc[county["model"].eq(model["model"]), "mean_signed_error"]
        np.testing.assert_allclose([values.min(), values.max()], model["county_bias_range"], atol=1e-8)

    boundaries = project / "code/data/吉林省.json"
    geo, repaired = read_geography(boundaries)
    assert set(annual["county"]) <= set(geo["name"])
    prefectures = geo.set_index("name")["prefecture"]
    annual.insert(3, "prefecture_adcode", annual["county"].map(prefectures))
    outer = geo.dissolve(by="prefecture").boundary
    moran = moran_table(county, graph, order)
    output.mkdir(parents=True, exist_ok=True)
    annual.to_csv(output / "county_year_errors.csv", index=False, encoding="utf-8-sig")
    county.to_csv(output / "county_mean_errors.csv", index=False, encoding="utf-8-sig")
    moran.to_csv(output / "moran_i.csv", index=False)

    figure = draw_page(geo, outer, county, order)
    figure.savefig(output / "figure4.pdf")
    figure.savefig(output / "figure4.png", dpi=600, facecolor="white", pil_kwargs={"optimize": True})
    plt.close(figure)
    with PdfPages(output / "annual_maps.pdf") as pdf:
        for year in YEARS:
            figure = draw_page(geo, outer, annual.loc[annual["year"].eq(year)], order, year)
            pdf.savefig(figure)
            plt.close(figure)

    meta = {"prediction_seed": SEED, "county_counts": YEARS,
            "prediction_sources": [{k: spec[k] for k in ("model", "source_kind", "prediction_file",
                                                        "prediction_sha256")} for spec in source["models"]],
            "figure4_source_sha256": sha256(FIGURE_SOURCE), "geojson_sha256": sha256(boundaries),
            "geometry_repaired_for_display": repaired, "display_geographic_crs": "EPSG:4326",
            "queen_edges_sha256": sha256(graph),
            "queen_source": "build_county_adjacency in runs/20261102_1543_box/source/data.py, verified against all five saved fold contexts",
            "queen_islands": sorted(ISLANDS), "figure4_limits": {"signed": [-3, 3], "absolute": [0, 3]},
            "annual_map_limits": {"signed": [-5, 5], "absolute": [0, 5]},
            "moran": {"permutations": PERMUTATIONS, "permutation_seed": SEED, "tail": "upper",
                      "weights": "row-standardized binary Queen adjacency, zero diagonal, islands excluded",
                      "adjustment": "Holm across six models"},
            "figure_dimensions_mm": [180, 190], "figure4_png_dpi": 600}
    (output / "provenance.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(moran.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PAPER.parent / "inputs")
    parser.add_argument("--queen-edges", type=Path, default=PAPER.parent / "inputs/queen_edges.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.project.resolve(), args.queen_edges.resolve(), args.output_dir.resolve())
