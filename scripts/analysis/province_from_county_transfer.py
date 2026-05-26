import argparse
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_year_value(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        y = int(value)
        return y if 1800 <= y <= 2200 else None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        y = int(value)
        return y if 1800 <= y <= 2200 else None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(18|19|20|21)\d{2}", s)
    return int(m.group(0)) if m else None


def _infer_col(df: pd.DataFrame, candidates, is_year=False, numeric=False) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if is_year:
        for c in cols:
            sample = df[c].dropna().head(20).tolist()
            if not sample:
                continue
            ok = sum(1 for v in sample if _parse_year_value(v) is not None)
            if ok >= max(3, int(0.6 * len(sample))):
                return c
    if numeric:
        best = None
        best_ratio = -1.0
        for c in cols:
            ratio = float(pd.to_numeric(df[c], errors="coerce").notna().mean())
            if ratio > best_ratio:
                best_ratio = ratio
                best = c
        return best
    return None


def load_county_truth(
    path: str,
    year_col: str,
    county_col: str,
    target_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    year_c = year_col if year_col else _infer_col(df, ["year", "年份", "时间"], is_year=True)
    county_c = county_col if county_col else _infer_col(df, ["name", "County", "县", "区县"])
    target_c = target_col if target_col else _infer_col(df, ["yield", "单产", "单产（公斤/公顷）"], numeric=True)
    if not (year_c and county_c and target_c):
        raise ValueError(f"Cannot infer county truth columns from {list(df.columns)}")
    out = df[[year_c, county_c, target_c]].copy()
    out.columns = ["year", "county", "y"]
    out["year"] = out["year"].apply(_parse_year_value)
    out["county"] = out["county"].astype(str)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["year", "county", "y"])
    out["year"] = out["year"].astype(int)
    return out


def load_county_pred(path: str, year_start: int, year_end: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"val_year", "county", "y_pred"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"County pred file missing columns {need}, got {list(df.columns)}")
    out = df[["val_year", "county", "y_pred"]].copy()
    out.columns = ["year", "county", "y"]
    out["year"] = out["year"].apply(_parse_year_value)
    out["county"] = out["county"].astype(str)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["year", "county", "y"])
    out["year"] = out["year"].astype(int)
    out = out[(out["year"] >= year_start) & (out["year"] <= year_end)]
    return out


def load_province_target(path: str, year_col: str, target_col: str, divisor: float) -> Dict[int, float]:
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    year_c = year_col if year_col else _infer_col(df, ["year", "年份", "时间", "Unnamed: 0"], is_year=True)
    target_c = target_col if target_col else _infer_col(df, ["单产（公斤/公顷）", "单产", "吉林"], numeric=True)
    if not (year_c and target_c):
        raise ValueError(f"Cannot infer province columns from {list(df.columns)}")
    d = float(divisor) if divisor else 1.0
    if d == 0:
        d = 1.0
    out: Dict[int, float] = {}
    for _, r in df[[year_c, target_c]].iterrows():
        y = _parse_year_value(r[year_c])
        v = pd.to_numeric(r[target_c], errors="coerce")
        if y is None or pd.isna(v):
            continue
        out[y] = float(v) / d
    return out


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, float]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    g = xc.T @ xc
    reg = float(alpha) * np.eye(g.shape[0], dtype=g.dtype)
    rhs = xc.T @ yc
    try:
        w = np.linalg.solve(g + reg, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(g + reg) @ rhs
    return w, x_mean, y_mean


def ridge_predict(x: np.ndarray, w: np.ndarray, x_mean: np.ndarray, y_mean: float) -> np.ndarray:
    return (x - x_mean) @ w + y_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-truth-file", default="data/Yield_Data.xlsx")
    parser.add_argument("--county-truth-year-col", default="year")
    parser.add_argument("--county-truth-county-col", default="name")
    parser.add_argument("--county-truth-target-col", default="yield")
    parser.add_argument("--county-preds-file", default="outputs/dual_tower/rollval_predictions.csv")
    parser.add_argument("--province-file", default="data/吉林省单产2000-2023.xlsx")
    parser.add_argument("--province-year-col", default="Unnamed: 0")
    parser.add_argument("--province-target-col", default="单产（公斤/公顷）")
    parser.add_argument("--province-target-divisor", type=float, default=1000.0)
    parser.add_argument("--train-end-year", type=int, default=2015)
    parser.add_argument("--predict-start-year", type=int, default=2016)
    parser.add_argument("--predict-end-year", type=int, default=2019)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--out-dir", default="outputs/dual_tower")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    county_true = load_county_truth(
        args.county_truth_file,
        args.county_truth_year_col,
        args.county_truth_county_col,
        args.county_truth_target_col,
    )
    county_pred = load_county_pred(args.county_preds_file, args.predict_start_year, args.predict_end_year)
    province_y = load_province_target(
        args.province_file,
        args.province_year_col,
        args.province_target_col,
        args.province_target_divisor,
    )

    train_true = county_true[county_true["year"] <= args.train_end_year].copy()
    if train_true.empty:
        raise RuntimeError("No county truth rows for training years")
    x_train_df = train_true.pivot_table(index="year", columns="county", values="y", aggfunc="mean").sort_index()
    x_test_df = county_pred.pivot_table(index="year", columns="county", values="y", aggfunc="mean").sort_index()
    if x_test_df.empty:
        raise RuntimeError("No county predicted rows in requested predict year range")

    common_cols = sorted(set(x_train_df.columns) & set(x_test_df.columns))
    if not common_cols:
        raise RuntimeError("No overlapping counties between county truth and county predictions")
    x_train = x_train_df[common_cols].to_numpy(dtype=np.float64).copy()
    x_test = x_test_df[common_cols].to_numpy(dtype=np.float64).copy()

    col_mean = np.nanmean(x_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    if np.isnan(x_train).any():
        idx = np.where(np.isnan(x_train))
        x_train[idx] = np.take(col_mean, idx[1])
    if np.isnan(x_test).any():
        idx = np.where(np.isnan(x_test))
        x_test[idx] = np.take(col_mean, idx[1])

    train_years = x_train_df.index.tolist()
    y_train = np.array([province_y.get(int(y), np.nan) for y in train_years], dtype=np.float64)
    keep = np.isfinite(y_train)
    x_train = x_train[keep]
    y_train = y_train[keep]
    if y_train.size == 0:
        raise RuntimeError("No province labels for training years")

    w, x_mean, y_mean = ridge_fit(x_train, y_train, args.ridge_alpha)
    y_pred = ridge_predict(x_test, w, x_mean, y_mean)

    out_rows = []
    test_years = [int(y) for y in x_test_df.index.tolist()]
    for i, year in enumerate(test_years):
        y_t = province_y.get(year, np.nan)
        y_p = float(y_pred[i])
        err = y_p - float(y_t) if np.isfinite(y_t) else np.nan
        out_rows.append(
            {
                "val_year": year,
                "y_true": float(y_t) if np.isfinite(y_t) else np.nan,
                "y_pred": y_p,
                "error": err,
                "n_train_years": int(y_train.size),
                "n_counties_used": int(len(common_cols)),
                "train_year_end": int(args.train_end_year),
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values("val_year")
    out_path = os.path.join(args.out_dir, "province_transfer_predictions.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    eval_df = out_df.dropna(subset=["y_true", "y_pred"])
    if not eval_df.empty:
        yt = eval_df["y_true"].to_numpy(dtype=np.float64)
        yp = eval_df["y_pred"].to_numpy(dtype=np.float64)
        e = yp - yt
        rmse = float(np.sqrt(np.mean(e ** 2)))
        mae = float(np.mean(np.abs(e)))
        mape = float(np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-6)) * 100.0)
        if yt.size > 1:
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        else:
            r2 = float("nan")
        print(f"Transfer eval years={len(eval_df)} RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.2f}% R2={r2:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
