import argparse
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_year_value(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        year = int(value)
        return year if 1800 <= year <= 2200 else None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        year = int(value)
        return year if 1800 <= year <= 2200 else None
    text = str(value).strip()
    if not text:
        return None
    m = re.search(r"(18|19|20|21)\d{2}", text)
    if not m:
        return None
    return int(m.group(0))


def load_province_targets(
    path: str,
    year_col: str,
    target_col: str,
    target_divisor: float,
) -> Dict[int, float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Province file not found: {path}")
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    cols = list(df.columns)

    year_col_eff = year_col if year_col and year_col in cols else None
    if year_col_eff is None:
        for c in cols:
            sample = df[c].dropna().head(20).tolist()
            if not sample:
                continue
            ok = sum(1 for v in sample if _parse_year_value(v) is not None)
            if ok >= max(3, int(0.6 * len(sample))):
                year_col_eff = c
                break
    if year_col_eff is None:
        raise ValueError(f"Cannot infer year column from {cols}")

    target_col_eff = target_col if target_col and target_col in cols else None
    if target_col_eff is None:
        best_col = None
        best_ratio = -1.0
        for c in cols:
            if c == year_col_eff:
                continue
            ratio = float(pd.to_numeric(df[c], errors="coerce").notna().mean())
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = c
        target_col_eff = best_col
    if target_col_eff is None:
        raise ValueError(f"Cannot infer target column from {cols}")

    divisor = float(target_divisor) if target_divisor else 1.0
    if divisor == 0:
        divisor = 1.0
    out: Dict[int, float] = {}
    for _, row in df[[year_col_eff, target_col_eff]].iterrows():
        year = _parse_year_value(row[year_col_eff])
        if year is None:
            continue
        val = pd.to_numeric(row[target_col_eff], errors="coerce")
        if pd.isna(val):
            continue
        out[year] = float(val) / divisor
    print(
        f"Loaded province targets: file={path}, year_col={year_col_eff}, "
        f"target_col={target_col_eff}, divisor={divisor:g}, years={len(out)}"
    )
    return out


def load_county_preds(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"County predictions file not found: {path}")
    df = pd.read_csv(path)
    required = {"val_year", "county", "y_pred"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"County preds missing columns {required}, got {list(df.columns)}")
    df = df.copy()
    df["val_year"] = df["val_year"].apply(_parse_year_value)
    df["county"] = df["county"].astype(str)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["val_year", "county", "y_pred"])
    df["val_year"] = df["val_year"].astype(int)
    return df


def yearly_county_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for year, g in df.groupby("val_year"):
        vals = g["y_pred"].to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        vals_sorted = np.sort(vals)
        k = max(1, int(math.ceil(0.2 * vals_sorted.size)))
        top20_mean = float(vals_sorted[-k:].mean())
        rows.append(
            {
                "year": int(year),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "top20_mean": top20_mean,
                "n_counties": int(vals.size),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    alpha: float,
) -> float:
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = float(y_train.mean())
    x_train_c = x_train - x_mean
    y_train_c = y_train - y_mean
    gram = x_train_c.T @ x_train_c
    reg = float(alpha) * np.eye(gram.shape[0], dtype=gram.dtype)
    rhs = x_train_c.T @ y_train_c
    try:
        w = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(gram + reg) @ rhs
    return float(((x_val - x_mean) @ w)[0] + y_mean)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-6)) * 100.0)
    if y_true.size > 1:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    else:
        r2 = float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--county-preds-file",
        default="outputs/dual_tower/rollval_predictions.csv",
        help="县级滚动预测CSV（需含 val_year/county/y_pred）",
    )
    parser.add_argument(
        "--province-file",
        default="data/吉林省单产2000-2023.xlsx",
        help="省级真实单产文件（csv/xlsx）",
    )
    parser.add_argument("--province-year-col", default="", help="省级年份列，留空自动识别")
    parser.add_argument("--province-target-col", default="", help="省级目标列，留空自动识别")
    parser.add_argument(
        "--province-target-divisor",
        type=float,
        default=1000.0,
        help="省级目标缩放除数（例如公斤/公顷转吨/公顷可设1000）",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="低维特征Ridge alpha")
    parser.add_argument("--out-dir", default="outputs/dual_tower", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    county_df = load_county_preds(args.county_preds_file)
    province_targets = load_province_targets(
        args.province_file,
        args.province_year_col,
        args.province_target_col,
        args.province_target_divisor,
    )

    stats_df = yearly_county_stats(county_df)
    if stats_df.empty:
        raise RuntimeError("No yearly county stats computed")
    stats_df["y_true"] = stats_df["year"].map(province_targets)
    stats_df = stats_df.dropna(subset=["y_true"]).copy()
    if stats_df.empty:
        raise RuntimeError("No overlapping years between county preds and province targets")

    feature_cols = ["mean", "median", "std", "min", "max", "top20_mean"]

    pred_rows: List[Dict[str, float]] = []
    years = sorted(stats_df["year"].astype(int).tolist())
    for year in years:
        val_row = stats_df[stats_df["year"] == year].iloc[0]
        y_true = float(val_row["y_true"])

        y_pred_simple_avg = float(val_row["mean"])

        train_hist = stats_df[stats_df["year"] < year]
        if train_hist.empty:
            y_pred_roll_avg = float("nan")
            y_pred_lowdim_ridge = float("nan")
            n_train_years = 0
        else:
            y_pred_roll_avg = float(train_hist["y_true"].mean())
            x_train = train_hist[feature_cols].to_numpy(dtype=np.float64)
            y_train = train_hist["y_true"].to_numpy(dtype=np.float64)
            x_val = val_row[feature_cols].to_numpy(dtype=np.float64).reshape(1, -1)
            y_pred_lowdim_ridge = ridge_fit_predict(x_train, y_train, x_val, args.ridge_alpha)
            n_train_years = int(train_hist.shape[0])

        pred_rows.append(
            {
                "val_year": int(year),
                "y_true": y_true,
                "pred_simple_avg": y_pred_simple_avg,
                "pred_roll_avg": y_pred_roll_avg,
                "pred_lowdim_ridge": y_pred_lowdim_ridge,
                "n_train_years": int(n_train_years),
                "n_counties": int(val_row["n_counties"]),
                "feat_mean": float(val_row["mean"]),
                "feat_median": float(val_row["median"]),
                "feat_std": float(val_row["std"]),
                "feat_min": float(val_row["min"]),
                "feat_max": float(val_row["max"]),
                "feat_top20_mean": float(val_row["top20_mean"]),
            }
        )

    pred_df = pd.DataFrame(pred_rows).sort_values("val_year")
    pred_out = os.path.join(args.out_dir, "province_baseline_predictions.csv")
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

    summary_rows = []
    for name, col in [
        ("simple_avg", "pred_simple_avg"),
        ("rolling_avg", "pred_roll_avg"),
        ("lowdim_ridge", "pred_lowdim_ridge"),
    ]:
        eval_df = pred_df.dropna(subset=["y_true", col])
        if eval_df.empty:
            continue
        m = metric_summary(
            eval_df["y_true"].to_numpy(dtype=np.float64),
            eval_df[col].to_numpy(dtype=np.float64),
        )
        summary_rows.append(
            {
                "baseline": name,
                "n_years": int(eval_df.shape[0]),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
                "r2": m["r2"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("baseline")
    summary_out = os.path.join(args.out_dir, "province_baseline_summary.csv")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print(f"Saved yearly predictions: {pred_out}")
    print(f"Saved summary metrics: {summary_out}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
