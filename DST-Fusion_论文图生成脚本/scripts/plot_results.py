import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _save_all_formats(fig, out_path, dpi):
    stem, _ = os.path.splitext(out_path)
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".eps", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".tiff", dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {stem}.pdf")
    print(f"Wrote {stem}.eps")
    print(f"Wrote {stem}.tiff")


def _read_results(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("model_name") or not row.get("val_year") or not row.get("best_val_rmse"):
                    continue
                r2_raw = row.get("best_val_r2", "")
                try:
                    best_val_r2 = float(r2_raw) if r2_raw != "" else float("nan")
                except ValueError:
                    best_val_r2 = float("nan")
                rows.append(
                    {
                        "model_name": row["model_name"].strip(),
                        "val_year": int(float(row["val_year"])),
                        "best_val_rmse": float(row["best_val_rmse"]),
                        "best_val_r2": best_val_r2,
                    }
                )
    return rows


def _aggregate_by_year(rows, model, metric_key):
    by_year = defaultdict(list)
    for r in rows:
        if r["model_name"] != model:
            continue
        value = r.get(metric_key, float("nan"))
        if np.isfinite(value):
            by_year[r["val_year"]].append(value)
    return {y: float(np.mean(v)) for y, v in by_year.items()}


def _display_tick_labels(labels):
    out = []
    for label in labels:
        if label in {"Transformer-daily", "LSTM-daily"}:
            out.append(label.replace("-", "\n"))
        elif label == "DST-Fusion":
            out.append("DST-\nFusion")
        else:
            out.append(label)
    return out


def _resolve_colors(models):
    model_color_map = {
        "single": "#2C6DA4",  # DDCN blue
        "rf": "#B9D5CB",  # pale mint
        "lstm_daily": "#F7E3A3",  # soft pale yellow
        "transformer_daily": "#5AAEB6",  # muted teal
        "stgcn": "#8E77C6",  # soft violet for extra baseline
        "graph": "#6BAF92",  # muted green
        "dual": "#6BAF92",  # DST-Fusion green
    }
    fallback = ["#2C6DA4", "#F7E3A3", "#5AAEB6", "#B9D5CB", "#6BAF92", "#8E77C6", "#B7A38B"]
    colors = []
    for i, model in enumerate(models):
        colors.append(model_color_map.get(model, fallback[i % len(fallback)]))
    return colors


def _bar_plot(means, stds, labels, colors, out_path, ylabel, dpi, value_fmt="{:.3f}"):
    x = np.arange(len(labels))
    tick_labels = _display_tick_labels(labels)
    fig, ax = plt.subplots(figsize=(190.0 / 25.4, 115.0 / 25.4))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        capsize=4,
        edgecolor="#333333",
        linewidth=1.0,
        width=0.72,
    )
    for i, b in enumerate(bars):
        if not np.isfinite(means[i]):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            value_fmt.format(means[i]),
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="#222222",
        )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=11.0, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11.0, width=1.0, direction="in")
    ax.set_ylabel(ylabel, fontsize=13.0, fontweight="bold")
    ax.set_xlabel("Model", fontsize=13.0, fontweight="bold")
    fig.tight_layout(pad=1.0)
    _save_all_formats(fig, out_path, dpi)
    plt.close(fig)


def _line_plot(series, labels, colors, out_path, ylabel, dpi):
    fig, ax = plt.subplots(figsize=(190.0 / 25.4, 115.0 / 25.4))
    all_years = []
    for (years, values), label, color in zip(series, labels, colors):
        ax.plot(
            years,
            values,
            marker="o",
            markersize=6.0,
            linewidth=2.2,
            color=color,
            label=label,
        )
        all_years.extend(years)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_xlabel("Year", fontsize=13.0, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13.0, fontweight="bold")
    ax.tick_params(labelsize=11.0, width=1.0, direction="in")
    if all_years:
        ticks = sorted(set(int(y) for y in all_years))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(y) for y in ticks])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=min(3, len(labels)),
        frameon=False,
        fontsize=10.5,
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=1.0)
    _save_all_formats(fig, out_path, dpi)
    plt.close(fig)


def _summarize_metric(rows, models, metric_key):
    means = []
    stds = []
    line_series = []
    has_any = False
    for model in models:
        year_means = _aggregate_by_year(rows, model, metric_key)
        if not year_means:
            means.append(float("nan"))
            stds.append(0.0)
            line_series.append(([], []))
            continue
        has_any = True
        years = sorted(year_means.keys())
        values = [year_means[y] for y in years]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
        line_series.append((years, values))
    return means, stds, line_series, has_any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-glob",
        default="outputs/*/results.csv",
        help="results.csv 路径通配符",
    )
    parser.add_argument(
        "--models",
        default="",
        help="模型名，逗号分隔，例如 single,rf",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="显示名，逗号分隔，例如 DDCN,RF",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/compare",
        help="输出目录",
    )
    parser.add_argument("--dpi", type=int, default=1000, help="TIFF export DPI")
    parser.add_argument(
        "--include-ablation",
        action="store_true",
        help="默认只绘制主模型和基线；加上此参数后才包含消融模型",
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="只绘制消融模型对比图",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.results_glob))
    if not paths:
        raise SystemExit("No results.csv files found.")
    rows = _read_results(paths)
    if not rows:
        raise SystemExit("No valid rows in results.csv.")

    model_alias = {
        "dual": "graph",
        "dst-fusion": "graph",
    }
    display_names = {
        "single": "DDCN",
        "rf": "RF",
        "lstm_daily": "LSTM-daily",
        "transformer_daily": "Transformer-daily",
        "stgcn": "STGCN",
        "graph": "DST-Fusion",
        "dual": "DST-Fusion",
        "dual_wo_graph": "DST-Fusion w/o graph",
        "graph_wo_sif": "DST-Fusion w/o SIF",
    }
    default_baseline_order = [
        "graph",
        "rf",
        "lstm_daily",
        "transformer_daily",
        "single",
        "stgcn",
    ]
    default_ablation_order = [
        "graph",
        "dual_wo_graph",
        "graph_wo_sif",
    ]

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        models = [model_alias.get(m.lower(), m) for m in models]
    else:
        found_models = {r["model_name"] for r in rows}
        if args.ablation_only:
            models = [m for m in default_ablation_order if m in found_models]
        elif args.include_ablation:
            models = [m for m in default_baseline_order if m in found_models]
            ablation_models = sorted(found_models - set(default_baseline_order))
            models.extend(ablation_models)
        else:
            models = [m for m in default_baseline_order if m in found_models]
    if len(models) < 2:
        raise SystemExit("Need at least two models to compare.")

    if args.labels:
        labels = [m.strip() for m in args.labels.split(",") if m.strip()]
    else:
        labels = [display_names.get(m, m) for m in models]
    if len(labels) < len(models):
        labels = labels + [display_names.get(m, m) for m in models[len(labels):]]
    labels = labels[: len(models)]

    colors = _resolve_colors(models)
    os.makedirs(args.out_dir, exist_ok=True)

    rmse_means, rmse_stds, rmse_line_series, has_rmse = _summarize_metric(
        rows, models, "best_val_rmse"
    )
    if has_rmse:
        rmse_bar_path = os.path.join(args.out_dir, "rolling_rmse_bar")
        rmse_line_path = os.path.join(args.out_dir, "yearly_rmse_line")
        _bar_plot(rmse_means, rmse_stds, labels, colors, rmse_bar_path, ylabel="RMSE", dpi=args.dpi)
        _line_plot(rmse_line_series, labels, colors, rmse_line_path, ylabel="RMSE", dpi=args.dpi)

    r2_means, r2_stds, r2_line_series, has_r2 = _summarize_metric(
        rows, models, "best_val_r2"
    )
    if has_r2:
        r2_bar_path = os.path.join(args.out_dir, "rolling_r2_bar")
        r2_line_path = os.path.join(args.out_dir, "yearly_r2_line")
        _bar_plot(r2_means, r2_stds, labels, colors, r2_bar_path, ylabel="R2", dpi=args.dpi, value_fmt="{:.4f}")
        _line_plot(r2_line_series, labels, colors, r2_line_path, ylabel="R2", dpi=args.dpi)
    else:
        print("No valid best_val_r2 found in results.csv; skipped R2 plots.")


if __name__ == "__main__":
    main()
