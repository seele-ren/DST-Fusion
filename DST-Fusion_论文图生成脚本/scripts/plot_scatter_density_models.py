import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


@dataclass
class ModelSpec:
    title: str
    csv_path: str
    results_path: str


def _read_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    y_true: List[float] = []
    y_pred: List[float] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                yt = float(row["y_true"])
                yp = float(row["y_pred"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(yt) and np.isfinite(yp):
                y_true.append(yt)
                y_pred.append(yp)
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def _point_density(x: np.ndarray, y: np.ndarray, bins: int = 120, bandwidth: float = 0.55) -> np.ndarray:
    if x.size == 0:
        return np.asarray([], dtype=np.float32)

    values = np.vstack([x, y]).astype(np.float64)
    n = values.shape[1]
    if n == 1:
        return np.ones(1, dtype=np.float32)

    cov = np.cov(values)
    cov = np.atleast_2d(cov)
    cov = cov + np.eye(2, dtype=np.float64) * 1e-6
    scale = max(float(bandwidth), 1e-3) ** 2
    cov = cov * scale
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    diffs = values[:, :, None] - values[:, None, :]
    quad = np.einsum("aij,ab,bij->ij", diffs, inv_cov, diffs, optimize=True)
    dens = np.exp(-0.5 * quad).sum(axis=1)
    dens = dens / max(float(np.max(dens)), 1e-12)
    return dens.astype(np.float32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if y_true.size == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return rmse, r2


def _metrics_from_results(path: str) -> Tuple[float, float]:
    rmse_vals: List[float] = []
    r2_vals: List[float] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def _nice_step(span: float) -> float:
    if span <= 4:
        return 1.0
    if span <= 8:
        return 2.0
    if span <= 15:
        return 2.5
    return 5.0


def _range_from_quantiles(values: List[np.ndarray], trim_q: float) -> Tuple[float, float]:
    arr = np.concatenate(values).astype(np.float64)
    q = min(max(float(trim_q), 0.0), 0.2)
    low = float(np.quantile(arr, q))
    high = float(np.quantile(arr, 1.0 - q))
    if high <= low:
        low = float(np.min(arr))
        high = float(np.max(arr))
    return low, high


def _save_all_formats(fig: plt.Figure, out_path: str, dpi: int) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    stem, _ = os.path.splitext(out_path)
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".eps", bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".tiff", dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {stem}.pdf")
    print(f"Wrote {stem}.eps")
    print(f"Wrote {stem}.tiff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model scatter plots with density coloring.")
    parser.add_argument("--rf", default="outputs/rf_baseline/rollval_predictions.csv")
    parser.add_argument("--lstm", default="outputs/lstm_daily_baseline/rollval_predictions.csv")
    parser.add_argument("--transformer", default="outputs/transformer_daily_baseline/rollval_predictions.csv")
    parser.add_argument("--stgcn", default="outputs/stgcn_baseline/rollval_predictions.csv")
    parser.add_argument("--single", default="outputs/single_tower/rollval_predictions.csv")
    parser.add_argument("--dual", default="outputs/dual_tower/rollval_predictions.csv")
    parser.add_argument("--rf-results", default="outputs/rf_baseline/results.csv")
    parser.add_argument("--lstm-results", default="outputs/lstm_daily_baseline/results.csv")
    parser.add_argument("--transformer-results", default="outputs/transformer_daily_baseline/results.csv")
    parser.add_argument("--stgcn-results", default="outputs/stgcn_baseline/results.csv")
    parser.add_argument("--single-results", default="outputs/single_tower/results.csv")
    parser.add_argument("--dual-results", default="outputs/dual_tower/results.csv")
    parser.add_argument("--out", default="outputs/compare/scatter_density_6models_paper")
    parser.add_argument("--title", default="", help="Figure title")
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=11.0)
    parser.add_argument("--point-alpha", type=float, default=1.0)
    parser.add_argument("--diag-offset", type=float, default=3.0, help="Offset for dashed lines around y=x")
    parser.add_argument("--tick-step", type=float, default=0.0, help="Major tick interval; <=0 means auto")
    parser.add_argument("--padding-ratio", type=float, default=0.03, help="Axis padding ratio")
    parser.add_argument("--bandwidth", type=float, default=0.55, help="Relative KDE bandwidth")
    parser.add_argument("--trim-quantile", type=float, default=0.01, help="Trim axis limits by quantiles to reduce whitespace")
    parser.add_argument("--fig-width-mm", type=float, default=190.0, help="Final figure width in mm")
    parser.add_argument("--fig-height-mm", type=float, default=138.0, help="Final figure height in mm")
    args = parser.parse_args()

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    specs = [
        ModelSpec("DST-Fusion", args.dual, args.dual_results),
        ModelSpec("RF", args.rf, args.rf_results),
        ModelSpec("LSTM-daily", args.lstm, args.lstm_results),
        ModelSpec("Transformer-daily", args.transformer, args.transformer_results),
        ModelSpec("DDCN", args.single, args.single_results),
        ModelSpec("STGCN", args.stgcn, args.stgcn_results),
    ]

    for spec in specs:
        if not os.path.exists(spec.csv_path):
            raise FileNotFoundError(f"Missing file: {spec.csv_path}")

    data = []
    all_true = []
    all_pred = []
    all_den = []
    for spec in specs:
        y_true, y_pred = _read_xy(spec.csv_path)
        den = _point_density(y_true, y_pred, bandwidth=args.bandwidth)
        metrics_results = None
        if os.path.exists(spec.results_path):
            metrics_results = _metrics_from_results(spec.results_path)
        data.append((spec, y_true, y_pred, den, metrics_results))
        if y_true.size:
            all_true.append(y_true)
            all_pred.append(y_pred)
        if den.size:
            all_den.append(den)

    if not all_true:
        raise SystemExit("No valid y_true/y_pred data found.")

    _, global_max = _range_from_quantiles(all_true + all_pred, args.trim_quantile)
    global_min = 0.0
    span = max(global_max - global_min, 1.0)
    pad = span * max(args.padding_ratio, 0.0)
    tick_step = float(args.tick_step) if args.tick_step and args.tick_step > 0 else _nice_step(span)
    lo = 0.0
    hi = math.ceil((global_max + pad) / tick_step) * tick_step
    if hi <= lo:
        hi = lo + tick_step
    ticks = np.arange(lo, hi + 0.5 * tick_step, tick_step)

    vmin = float(min(np.min(d) for d in all_den)) if all_den else 0.0
    vmax = float(max(np.max(d) for d in all_den)) if all_den else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)

    n_models = len(specs)
    ncols = 3
    nrows = 2
    fig_width_in = args.fig_width_mm / 25.4
    fig_height_in = args.fig_height_mm / 25.4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width_in, fig_height_in),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    if args.title:
        fig.suptitle(args.title, y=0.985, fontsize=15, fontweight="bold")
    cmap = plt.get_cmap("turbo")
    sm = None

    for idx, (ax, (spec, y_true, y_pred, den, metrics_results)) in enumerate(zip(axes, data)):
        if y_true.size == 0:
            ax.set_title(spec.title, fontsize=13.2, fontweight="bold")
            ax.set_axis_off()
            continue

        order = np.argsort(den)
        y_true_s = y_true[order]
        y_pred_s = y_pred[order]
        den_s = den[order]

        sc = ax.scatter(
            y_true_s,
            y_pred_s,
            c=den_s,
            cmap=cmap,
            norm=norm,
            s=args.point_size,
            alpha=args.point_alpha,
            edgecolors="none",
            rasterized=True,
        )
        sm = sc

        ax.plot([lo, hi], [lo, hi], color="#d81f1f", linewidth=1.9, zorder=1)
        off = float(args.diag_offset)
        ax.plot([lo, hi], [lo + off, hi + off], color="#111111", linewidth=1.55, linestyle="--", dashes=(3.5, 2.2), zorder=1)
        ax.plot([lo, hi], [lo - off, hi - off], color="#111111", linewidth=1.55, linestyle="--", dashes=(3.5, 2.2), zorder=1)

        ax.set_xlim(lo, hi, auto=False)
        ax.set_ylim(lo, hi, auto=False)
        ax.margins(x=0.0, y=0.0)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{int(t)}" if float(t).is_integer() else f"{t:g}" for t in ticks])
        ax.set_yticklabels([f"{int(t)}" if float(t).is_integer() else f"{t:g}" for t in ticks])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(spec.title, fontsize=13.2, fontweight="bold", pad=6)
        ax.tick_params(
            direction="in",
            length=3.0,
            width=1.05,
            labelsize=10.6,
            pad=1.5,
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
            labelleft=True,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)
            spine.set_color("#4a4a4a")

        row = idx // ncols
        col = idx % ncols
        if col == 0:
            ax.set_ylabel("Estimated yield (Mg ha$^{-1}$)", fontsize=12.6, fontweight="bold", labelpad=1.8)
        else:
            ax.set_ylabel("")
        if row == nrows - 1:
            ax.set_xlabel("Actual yield (Mg ha$^{-1}$)", fontsize=12.4, fontweight="bold", labelpad=1.6)
        else:
            ax.set_xlabel("")

        if metrics_results is not None and np.isfinite(metrics_results[0]) and np.isfinite(metrics_results[1]):
            rmse, r2 = metrics_results
        else:
            rmse, r2 = _metrics(y_true, y_pred)
        ax.text(
            0.965,
            0.07,
            f"RMSE: {rmse:.2f}\nR$^2$: {r2:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.6},
        )

    if sm is not None:
        cax = fig.add_axes([0.365, 0.012, 0.27, 0.018])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=10.0, direction="in", length=2.3, width=0.9)
        cbar.set_label("Density of data points", fontsize=12.6, fontweight="bold", labelpad=6)
        cbar.outline.set_linewidth(1.0)

    fig.subplots_adjust(left=0.06, right=0.992, top=0.955, bottom=0.205, wspace=0.11, hspace=0.24)
    _save_all_formats(fig, args.out, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
