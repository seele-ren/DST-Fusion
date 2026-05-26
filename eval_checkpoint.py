"""
评估脚本：加载训练好的模型权重，在测试年份上计算指标并绘制散点图。
"""
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import os
import matplotlib
# 无界面后端，便于在服务器环境保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文显示设置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from data import ArrayYieldDataset, CsvYieldDataset
from model import DDCN, DualTowerDDCN


def _parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    """把逗号分隔的字符串解析成整数列表。"""
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in items] if items else None


def _load_checkpoint(path: str) -> dict:
    """读取模型权重文件（.pt）。"""
    return torch.load(path, map_location="cpu", weights_only=False)


def _infer_dual_tower(ckpt: dict) -> bool:
    """从权重字段名推断是否是双塔模型。"""
    model_state = ckpt.get("model", {})
    if isinstance(model_state, dict):
        for key in model_state.keys():
            if key.startswith("dynamic_tower.") or key.startswith("context_tower.") or key.startswith("gate_net."):
                return True
    return False


def _build_model(ckpt: dict) -> Tuple[torch.nn.Module, bool]:
    """根据 checkpoint 中的参数构建模型结构。"""
    model_args = ckpt.get("model_args", {})
    use_static = bool(ckpt.get("use_static", model_args.get("use_static", False)))
    if not use_static:
        use_static = _infer_dual_tower(ckpt)
    if use_static:
        # 双塔模型：动态时序 + 静态特征
        model = DualTowerDDCN(
            feature_dim=model_args.get("feature_dim", 9),
            static_dim=model_args.get("static_dim", 3),
            sif_week_idx=model_args.get("sif_week_idx"),
            sif_missing_idx=model_args.get("sif_missing_idx"),
            use_week_extra_concat=model_args.get("use_week_extra_concat", False),
            week_extra_dim=model_args.get("week_extra_dim", 0),
            week_dim=model_args.get("week_dim", 128),
            transformer_heads=model_args.get("transformer_heads", 2),
            stage_hidden=model_args.get("stage_hidden", 256),
            season_hidden=model_args.get("season_hidden", 256),
            tower_output_dim=model_args.get("tower_output_dim", 128),
            lstm_layers=model_args.get("lstm_layers", 1),
            stage_lengths=model_args.get("stage_lengths", [5, 5, 5, 4, 6]),
            stage_ratio=model_args.get("stage_ratio"),
            fusion_hidden=model_args.get("fusion_hidden", 256),
            dropout=model_args.get("dropout", 0.2112762832922651),
        )
    else:
        # 单塔模型：仅动态时序
        model = DDCN(
            feature_dim=model_args.get("feature_dim", 9),
            sif_week_idx=model_args.get("sif_week_idx"),
            sif_missing_idx=model_args.get("sif_missing_idx"),
            use_week_extra_concat=model_args.get("use_week_extra_concat", False),
            week_extra_dim=model_args.get("week_extra_dim", 0),
            week_dim=model_args.get("week_dim", 128),
            transformer_heads=model_args.get("transformer_heads", 2),
            stage_hidden=model_args.get("stage_hidden", 256),
            season_hidden=model_args.get("season_hidden", 256),
            lstm_layers=model_args.get("lstm_layers", 1),
            stage_lengths=model_args.get("stage_lengths", [5, 5, 5, 4, 6]),
            stage_ratio=model_args.get("stage_ratio"),
            dropout=model_args.get("dropout", 0.2112762832922651),
        )
    return model, use_static


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
    use_static: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """?????????????????????"""
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        weekly = None
        if use_static:
            if len(batch) == 4:
                x, y, weekly, static = batch
            else:
                x, y, static = batch
            x = x.to(device)
            y = y.to(device)
            if weekly is not None:
                weekly = weekly.to(device)
            static = static.to(device)
            pred = model(x, static, weekly) if weekly is not None else model(x, static)
        else:
            if len(batch) == 3:
                x, y, weekly = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
            if weekly is not None:
                weekly = weekly.to(device)
            pred = model(x, weekly) if weekly is not None else model(x)
        # ???????????
        pred_raw = pred * y_std + y_mean
        y_raw = y * y_std + y_mean
        preds.append(pred_raw.detach().cpu())
        targets.append(y_raw.detach().cpu())
    preds = torch.cat(preds, dim=0).view(-1).numpy()
    targets = torch.cat(targets, dim=0).view(-1).numpy()
    return preds, targets


def _metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """计算常用评估指标，返回字典。"""
    diff = preds - targets
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    mean_true = float(np.mean(targets))
    mean_pred = float(np.mean(preds))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((targets - mean_true) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    nrmse = float("nan") if abs(mean_true) < 1e-6 else float(rmse / abs(mean_true))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse": nrmse,
        "mean_true": mean_true,
        "mean_pred": mean_pred,
        "mean_diff": mean_pred - mean_true,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "std_true": float(np.std(targets)),
    }


def main() -> None:
    """脚本入口：加载模型、准备数据、评估并绘图。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="模型权重路径（.pt）")
    parser.add_argument("--data", default="data/All_Data.csv", help="日尺度 CSV 数据")
    parser.add_argument("--yield-csv", default="data/Yield_Data.xlsx", help="产量文件（.csv 或 .xlsx）")
    parser.add_argument("--yield-col", default="yield")
    parser.add_argument("--yield-county-col", default="name")
    parser.add_argument("--yield-year-col", default="year")
    parser.add_argument(
        "--feature-cols",
        default="WDRVI_median,GCI_median,EVI_median,NDWI_median,NIRv_median,GDD,KDD,PRCP,VPD",
    )
    parser.add_argument("--target-col", default="yield")
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--county-col", default="County")
    parser.add_argument("--start-doy", type=int, default=128)
    parser.add_argument("--end-doy", type=int, default=305)
    parser.add_argument("--test-year-start", type=int, default=2018)
    parser.add_argument("--test-year-end", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--static-csv", default="data/jilin_phenology_2001-2020.csv")
    parser.add_argument("--static-cols", default="SOS,VGS,RGS")
    parser.add_argument("--fill-missing", choices=["ffill", "zero"], default="ffill")
    parser.add_argument("--out-dir", default="outputs", help="输出目录（图表）")
    args = parser.parse_args()

    # 1) 读取权重并构建模型
    ckpt = _load_checkpoint(args.checkpoint)
    model, use_static = _build_model(ckpt)
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device(args.device)
    model.to(device)

    # 2) 取出训练时的标准化统计量
    x_mean = ckpt.get("x_mean")
    x_std = ckpt.get("x_std")
    y_mean = ckpt.get("y_mean")
    y_std = ckpt.get("y_std")
    weekly_mean = ckpt.get("weekly_mean")
    weekly_std = ckpt.get("weekly_std")
    static_mean = ckpt.get("static_mean")
    static_std = ckpt.get("static_std")

    # 3) 准备数据集（按训练时的统计量标准化）
    cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    static_cols = [c.strip() for c in args.static_cols.split(",") if c.strip()] if use_static else None

    ds = CsvYieldDataset(
        args.data,
        feature_cols=cols,
        target_col=args.target_col,
        date_col=args.date_col,
        county_col=args.county_col,
        yield_csv=args.yield_csv,
        yield_col=args.yield_col,
        yield_county_col=args.yield_county_col,
        yield_year_col=args.yield_year_col,
        start_doy=args.start_doy,
        end_doy=args.end_doy,
        fill_missing=args.fill_missing,
        weekly_mean=weekly_mean,
        weekly_std=weekly_std,
        mean=x_mean,
        std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        static_cols=static_cols,
        static_csv=args.static_csv if use_static else None,
        static_mean=static_mean,
        static_std=static_std,
    )

    # 4) 根据年份切出测试集
    test_mask = (ds.years >= args.test_year_start) & (ds.years <= args.test_year_end)
    if test_mask.sum() == 0:
        raise ValueError("测试集年份范围内没有样本")

    # 5) 取出测试数据，并根据是否双塔准备静态特征
    x_test = ds.x[test_mask]
    y_test = ds.y[test_mask]
    static_test = None
    static_mask_test = None
    if use_static:
        if ds.static is None:
            raise ValueError("静态特征缺失，无法评估双塔模型")
        static_test = ds.static[test_mask]
        static_mask_test = ds.static_mask[test_mask] if ds.static_mask is not None else None

    # 6) 构建 DataLoader
    test_ds = ArrayYieldDataset(
        x=x_test,
        y=y_test,
        static=static_test,
        static_mask=static_mask_test,
        years=ds.years[test_mask],
        mean=x_mean,
        std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        weekly_mean=weekly_mean,
        weekly_std=weekly_std,
        static_mean=static_mean,
        static_std=static_std,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # 7) 预测与计算指标
    preds, targets = _predict(model, test_loader, device, y_mean, y_std, use_static)
    metrics = _metrics(preds, targets)

    print(f"测试集 {args.test_year_start}-{args.test_year_end} 样本数 {targets.shape[0]}")
    print(
        "RMSE {rmse:.4f} MAE {mae:.4f} NRMSE {nrmse:.4f} R2 {r2:.4f}".format(**metrics)
    )
    print("R2 分解: ss_res {ss_res:.4f} ss_tot {ss_tot:.4f} std_true {std_true:.4f}".format(**metrics))
    preview_count = min(5, targets.shape[0])
    if preview_count > 0:
        print("前几个样本真实值 预测值:")
        for i in range(preview_count):
            print(f"{targets[i]:.4f}, {preds[i]:.4f}")
    print(
        "真实均值 {mean_true:.4f} 预测均值 {mean_pred:.4f} 均值差 {mean_diff:.4f}".format(
            **metrics
        )
    )
    # 8) 绘制散点图
    os.makedirs(args.out_dir, exist_ok=True)
    scatter_path = os.path.join(args.out_dir, "eval_test_scatter.png")
    plt.figure(figsize=(6, 6))
    print(f"开始绘制测试集散点图，共 {len(targets)} 个样本")
    test_years = ds.years[test_mask]
    for yr in (2018, 2019):
        yr_mask = test_years == yr
        if yr_mask.any():
            yr_metrics = _metrics(preds[yr_mask], targets[yr_mask])
            print(
                f"{yr}年 R2 {yr_metrics['r2']:.4f} RMSE {yr_metrics['rmse']:.4f} "
                f"样本 {int(yr_mask.sum())}"
            )
            if yr == 2019:
                print(
                    f"{yr}年 真实均值 {yr_metrics['mean_true']:.4f} 预测均值 {yr_metrics['mean_pred']:.4f}"
                )
                pred_std = float(np.std(preds[yr_mask]))
                true_std = float(np.std(targets[yr_mask]))
                print(f"{yr}年 真实标准差 {true_std:.4f} 预测标准差 {pred_std:.4f}")
                if yr_mask.sum() >= 2:
                    corr = float(np.corrcoef(targets[yr_mask], preds[yr_mask])[0, 1])
                    print(f"{yr}年 Pearson r {corr:.4f}")
    mask_2018 = test_years == 2018
    mask_2019 = test_years == 2019
    other_mask = ~(mask_2018 | mask_2019)
    if mask_2018.any():
        plt.scatter(
            targets[mask_2018],
            preds[mask_2018],
            s=14,
            alpha=0.7,
            edgecolors="none",
            label="2018年",
            color="#1f77b4",
        )
    if mask_2019.any():
        plt.scatter(
            targets[mask_2019],
            preds[mask_2019],
            s=14,
            alpha=0.7,
            edgecolors="none",
            label="2019年",
            color="#ff7f0e",
        )
    if other_mask.any():
        plt.scatter(
            targets[other_mask],
            preds[other_mask],
            s=12,
            alpha=0.5,
            edgecolors="none",
            label="其他年份",
            color="#7f7f7f",
        )
    min_val = float(min(targets.min(), preds.min()))
    max_val = float(max(targets.max(), preds.max()))
    padding = (max_val - min_val) * 0.05
    line_x = np.linspace(min_val - padding, max_val + padding, 200)
    plt.plot(line_x, line_x, color="black", linewidth=1.2, label="1:1 对角线")
    lower = line_x * 0.9
    upper = line_x * 1.1
    plt.fill_between(line_x, lower, upper, color="#d9d9d9", alpha=0.35, label="±10% 区域")
    if targets.size >= 2:
        slope, intercept = np.polyfit(targets, preds, 1)
        reg_y = slope * line_x + intercept
        plt.plot(line_x, reg_y, color="#2ca02c", linewidth=1.2, label="回归趋势线")
    plt.xlabel("真实产量")
    plt.ylabel("预测产量")
    plt.title("模型测试散点图")
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"散点图已保存到 {os.path.abspath(scatter_path)}")


if __name__ == "__main__":
    main()
