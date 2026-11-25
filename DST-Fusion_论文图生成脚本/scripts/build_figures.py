"""Rebuild the current DST-Fusion manuscript figure files from bundled inputs."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


SCRIPTS = Path(__file__).resolve().parent
INPUTS = SCRIPTS.parent / "inputs"
METRICS = INPUTS / "code/outputs"
COUNTY = INPUTS / "code/data/吉林省.json"
SCATTER_MODELS = {
    "dual": "dual_tower",
    "rf": "rf_baseline",
    "lstm": "lstm_daily_baseline",
    "transformer": "transformer_daily_baseline",
    "single": "single_tower",
    "stgcn": "stgcn_baseline",
}
FIGURES = ("1", "3", "4", "5", "appendix", "prior")


def run(script, *args):
    subprocess.run(
        [sys.executable, "-B", str(SCRIPTS / script), *map(str, args)],
        check=True, env={**os.environ, "MPLBACKEND": "Agg"},
    )


def build_study_area(output, ccd):
    if ccd:
        mask_args = ["--ccd-maize", ccd, "--save-display-mask", output / "study_area_display_mask.tif"]
    else:
        mask_args = ["--display-mask", INPUTS / "study_area/maize_occurrence_2001_2019.tif"]
    run(
        "plot_submission_study_area.py", *mask_args,
        "--boundaries", COUNTY, "--predictions", METRICS / "dual_tower/rollval_predictions.csv",
        "--pdf", output / "ren1.pdf", "--png", output / "ren1.png",
    )


def build_metrics(output, scratch, selected):
    work = scratch / "metrics"
    run("plot_results.py", "--results-glob", str(METRICS / "*/results.csv"), "--out-dir", work, "--dpi", 120)
    panels = {
        "3": {"rolling_rmse_bar": "ren3", "rolling_r2_bar": "ren4"},
        "5": {"yearly_rmse_line": "ren7", "yearly_r2_line": "ren8"},
    }
    for number in ("3", "5"):
        if number in selected:
            for source, target in panels[number].items():
                shutil.copyfile(work / f"{source}.pdf", output / f"{target}.pdf")


def build_scatter(output, scratch):
    args = []
    for option, model in SCATTER_MODELS.items():
        args.extend([f"--{option}", METRICS / model / "rollval_predictions.csv"])
        args.extend([f"--{option}-results", METRICS / model / "results.csv"])
    run("plot_scatter_density_models.py", *args, "--out", scratch / "ren5", "--dpi", 120)
    shutil.copyfile(scratch / "ren5.pdf", output / "ren5.pdf")


def build_prior(output, scratch):
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((SCRIPTS.parent / "provenance/prior_ren6_input_sources.json").read_text())
    map_inputs = scratch / "prior_map_inputs"
    for model, source in sources.items():
        dest = map_inputs / model
        dest.mkdir(parents=True)
        shutil.copyfile(INPUTS / source["prediction"], dest / "rollval_predictions.csv")
        shutil.copyfile(INPUTS / source["annotation_summary"], dest / "results.csv")
    run(
        "plot_choropleth_models.py", "--geojson", COUNTY, "--base-out", map_inputs,
        "--out-dir", scratch / "prior_map", "--out-name", "ren6", "--dpi", 120,
    )
    shutil.copyfile(scratch / "prior_map/ren6.pdf", output / "ren6.pdf")

    ablation_inputs = scratch / "prior_ablation_inputs"
    ablation_models = {
        "graph": "dual_tower",
        "dual_wo_graph": "outputs_ablation_wo_graph/dual_tower",
        "graph_wo_sif": "outputs_ablation_wo_sif/dual_tower",
    }
    for model, source in ablation_models.items():
        target = ablation_inputs / model
        target.mkdir(parents=True)
        with (METRICS / source / "results.csv").open(encoding="utf-8-sig", newline="") as dataset, (
            target / "results.csv"
        ).open("w", encoding="utf-8", newline="") as results:
            reader = csv.DictReader(dataset)
            writer = csv.DictWriter(results, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow({**row, "model_name": model})
    ablation = scratch / "prior_ablation"
    run(
        "plot_results.py", "--results-glob", str(ablation_inputs / "*/results.csv"),
        "--ablation-only", "--out-dir", ablation, "--dpi", 120,
    )
    for source, target in (("rolling_rmse_bar", "ren9"), ("yearly_rmse_line", "ren10")):
        shutil.copyfile(ablation / f"{source}.pdf", output / f"{target}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=FIGURES[:-1])
    parser.add_argument("--ccd-maize", type=Path, help="Optionally rebuild the Figure 1 layer from annual maps.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.figures)
    with TemporaryDirectory(prefix="dstfusion-figures-") as temporary:
        scratch = Path(temporary)
        if "1" in selected:
            build_study_area(output, args.ccd_maize)
        if selected & {"3", "5"}:
            build_metrics(output, scratch, selected)
        if "3" in selected:
            build_scatter(output, scratch)
        if selected & {"4", "appendix"}:
            run("county_error_maps/analyze_and_plot.py", "--output-dir", output / "county_error_maps")
        if "prior" in selected:
            build_prior(output / "prior", scratch)


if __name__ == "__main__":
    main()
