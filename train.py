"""
训练脚本：从县域/年份的作物时序特征中预测产量。

面向非专业读者的阅读提示：
- 数据读取：把 CSV 的逐日/逐周特征整理成模型输入张量。
- 统计量：训练集用于计算均值/标准差，用于标准化与反标准化。
- 模型训练：DDCN 或双塔 DDCN（动态+静态特征）。
- 评估与保存：输出 RMSE/NRMSE/R2，并保存模型权重与统计量。
"""
import argparse
import csv
import glob
import math
import os
import random
import sys
from copy import deepcopy
from typing import Optional, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
# 使用无界面后端，适合服务器/批处理环境保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

# 中文字体设置，避免图中中文显示为方块
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from data import (
    ArrayYieldDataset,
    CsvYieldDataset,
    GraphYearDataset,
    build_county_adjacency,
    build_graph_year_samples,
    compute_feature_stats,
    compute_target_stats,
    compute_static_stats,
    expand_static_cols,
    load_province_year_targets,
    normalize_features,
)
from model import DDCN, DualTowerDDCN, DSTfusion


def set_seed(seed: int) -> None:
    """固定随机种子，保证多次运行结果尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 下面两行让 CUDA 的计算更稳定，但可能略微降低速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(
    train_path: str,
    val_path: Optional[str],
    batch_size: int,
    num_workers: int,
    feature_cols: Optional[str],
    target_col: str,
    date_col: str,
    county_col: str,
    yield_csv: Optional[str],
    yield_col: str,
    yield_county_col: str,
    yield_year_col: str,
    start_doy: int,
    end_doy: int,
    fill_missing: str,
) -> Tuple[DataLoader, Optional[DataLoader], Tuple]:
    """
    从 CSV 构建训练/验证 DataLoader，并返回用于标准化的统计量。

    返回：
    - train_loader: 训练批次迭代器
    - val_loader: 可选验证集迭代器
    - stats: (x_mean, x_std, y_mean, y_std)
    """
    if not train_path.lower().endswith(".csv"):
        raise ValueError("仅支持 CSV 训练集")
    if not feature_cols:
        raise ValueError("CSV 输入必须提供特征列名")
    cols = [c.strip() for c in feature_cols.split(",") if c.strip()]
    # CSV 训练集：先构建数据，再计算标准化统计量
    train_ds = CsvYieldDataset(
        train_path,
        feature_cols=cols,
        target_col=target_col,
        date_col=date_col,
        county_col=county_col,
        yield_csv=yield_csv,
        yield_col=yield_col,
        yield_county_col=yield_county_col,
        yield_year_col=yield_year_col,
        start_doy=start_doy,
        end_doy=end_doy,
        fill_missing=fill_missing,
    )
    x_mean, x_std = compute_feature_stats(train_ds.x)
    y_mean, y_std = compute_target_stats(train_ds.y)
    train_ds.mean = x_mean
    train_ds.std = x_std
    train_ds.y_mean = y_mean
    train_ds.y_std = y_std
    train_ds._apply_tensor_cache()

    # 训练集打乱，有利于收敛
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if val_path:
        if not val_path.lower().endswith(".csv"):
            raise ValueError("仅支持 CSV 验证集")
        cols = [c.strip() for c in (feature_cols or "").split(",") if c.strip()]
        # 验证集使用训练集统计量（避免数据泄漏）
        val_ds = CsvYieldDataset(
            val_path,
            feature_cols=cols,
            target_col=target_col,
            date_col=date_col,
            county_col=county_col,
            yield_csv=yield_csv,
            yield_col=yield_col,
            yield_county_col=yield_county_col,
            yield_year_col=yield_year_col,
            start_doy=start_doy,
            end_doy=end_doy,
            fill_missing=fill_missing,
            mean=x_mean,
            std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )
        # 验证集不打乱，便于稳定评估
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    stats = (x_mean, x_std, y_mean, y_std)
    return train_loader, val_loader, stats


def build_array_loader(
    x,
    y,
    years,
    batch_size,
    shuffle,
    num_workers,
    x_mean,
    x_std,
    y_mean,
    y_std,
    weekly=None,
    static=None,
    static_mask=None,
    static_mean=None,
    static_std=None,
):
    """
    将内存中的 numpy/torch 数组封装为 DataLoader。

    适用于已按年份切分好的数据，避免重复读取磁盘。
    """
    ds = ArrayYieldDataset(
        x=x,
        y=y,
        weekly=weekly,
        static=static,
        static_mask=static_mask,
        years=years,
        mean=x_mean,
        std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        static_mean=static_mean,
        static_std=static_std,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _get_stage_params(args):
    """解析阶段长度与比例的命令行参数。"""
    stage_lengths_list = [int(x) for x in args.stage_lengths.split(",")]
    stage_ratio_list = [int(x) for x in args.stage_ratio.split(",")] if args.stage_ratio else None
    return stage_lengths_list, stage_ratio_list


def _resolve_col_indices(all_cols, selected_cols, label):
    if not selected_cols:
        return []
    missing = [c for c in selected_cols if c not in all_cols]
    if missing:
        raise ValueError(f"{label} 列在特征列表中不存在: {missing}")
    return [all_cols.index(c) for c in selected_cols]


def _resolve_first_existing_index(all_cols, candidates, label, required=False):
    for name in candidates:
        if name and name in all_cols:
            return all_cols.index(name)
    if required:
        raise ValueError(f"{label} 列在特征列表中不存在: {candidates}")
    return None


def _find_first_col(all_cols, include_keywords, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    for col in all_cols:
        lowered = col.lower()
        if all(key in lowered for key in include_keywords) and not any(
            key in lowered for key in exclude_keywords
        ):
            return col
    return None


def get_model_args(args, use_static, overrides=None):
    """
    生成模型初始化所需的参数字典，便于保存/复现实验。
    """
    stage_lengths_list, stage_ratio_list = _get_stage_params(args)
    model_args = {
        "feature_dim": getattr(args, "feature_dim", 9),
        "week_dim": args.week_dim,
        "transformer_heads": args.transformer_heads,
        "stage_hidden": args.stage_hidden,
        "season_hidden": args.season_hidden,
        "lstm_layers": args.lstm_layers,
        "stage_lengths": stage_lengths_list,
        "stage_ratio": stage_ratio_list,
        "dropout": args.dropout,
        "sif_week_idx": getattr(args, "sif_week_idx", None),
        "sif_missing_idx": getattr(args, "sif_missing_idx", None),
        "week_extra_dim": getattr(args, "week_extra_dim", 0),
        "use_static": bool(use_static),
        "use_gnn": bool(getattr(args, "use_gnn", False)),
    }
    if use_static:
        model_args.update({
            "static_dim": getattr(args, "static_dim", 3),
            "tower_output_dim": getattr(args, "tower_output_dim", 128),
            "fusion_hidden": getattr(args, "fusion_hidden", 256),
            "meteo_idx": getattr(args, "meteo_idx", []),
            "gnn_hidden": getattr(args, "gnn_hidden", 128),
            "gnn_layers": getattr(args, "gnn_layers", 2),
            "geojson": getattr(args, "geojson", None),
            "static_cols": getattr(args, "static_cols", None),
            "gnn_alpha0": getattr(args, "gnn_alpha0", 0.1),
            "gnn_edge_dropout": getattr(args, "gnn_edge_dropout", 0.0),
            "enable_province_task": bool(getattr(args, "enable_province_task", False)),
            "province_loss_weight": getattr(args, "province_loss_weight", 1.0),
        })
    if overrides:
        model_args.update(overrides)
    return model_args


def _compute_stats_for_mask(x_all, y_all, static_all, static_mask_all, mask, use_static):
    """
    对指定 mask 的子集计算均值/标准差。

    用于滚动验证或固定划分场景，保证只用训练部分的统计量。
    """
    x_mean, x_std = compute_feature_stats(x_all[mask])
    y_mean, y_std = compute_target_stats(y_all[mask])
    static_mean = None
    static_std = None
    if use_static and static_all is not None:
        static_mean, static_std = compute_static_stats(
            static_all[mask],
            static_mask_all[mask] if static_mask_all is not None else None,
        )
    return x_mean, x_std, y_mean, y_std, static_mean, static_std


def _build_array_loader_for_mask(
    x_all,
    y_all,
    weekly_all,
    years,
    mask,
    batch_size,
    shuffle,
    num_workers,
    stats,
    static_all,
    static_mask_all,
    use_static,
):
    """根据 mask 构建 DataLoader，并带上对应统计量用于标准化。"""
    x_mean, x_std, y_mean, y_std, static_mean, static_std = stats
    return build_array_loader(
        x_all[mask],
        y_all[mask],
        years[mask],
        batch_size,
        shuffle,
        num_workers,
        x_mean,
        x_std,
        y_mean,
        y_std,
        weekly=weekly_all[mask] if weekly_all is not None else None,
        static=static_all[mask] if use_static and static_all is not None else None,
        static_mask=static_mask_all[mask] if use_static and static_mask_all is not None else None,
        static_mean=static_mean,
        static_std=static_std,
    )


def _build_graph_loader_for_mask(
    x_all,
    y_all,
    weekly_all,
    years,
    counties,
    mask,
    batch_size,
    shuffle,
    num_workers,
    stats,
    static_all,
    county_order,
    adj,
    static_mean=None,
    static_std=None,
    province_targets=None,
    samples=None,
    sample_years=None,
):
    """根据 mask 构建 GraphYearDataset DataLoader。"""
    x_mean, x_std, y_mean, y_std, static_mean_masked, static_std_masked = stats
    if static_mean is None:
        static_mean = static_mean_masked
    if static_std is None:
        static_std = static_std_masked
    ds = GraphYearDataset(
        x_all[mask],
        y_all[mask],
        years[mask],
        counties[mask],
        county_order,
        adj,
        weekly=weekly_all[mask] if weekly_all is not None else None,
        static=static_all[mask] if static_all is not None else None,
        mean=x_mean,
        std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        static_mean=static_mean,
        static_std=static_std,
        province_targets=province_targets,
        samples=samples,
        sample_years=sample_years,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def save_checkpoint(path, model_state, stats, use_static, model_args, extra=None):
    """
    保存模型权重和标准化统计量，便于后续加载预测。
    """
    x_mean, x_std, y_mean, y_std, static_mean, static_std = stats
    save_dict = {
        "model": model_state,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "use_static": bool(use_static),
        "model_args": model_args,
    }
    if use_static and static_mean is not None:
        save_dict["static_mean"] = static_mean
        save_dict["static_std"] = static_std
    if extra:
        save_dict.update(extra)
    torch.save(save_dict, path)


def print_dataset_stats(name, dataset):
    """打印数据集的基本统计信息，帮助检查数据是否合理。"""
    if hasattr(dataset, "stats"):
        s = dataset.stats
        print(
            f"{name} 数据集统计: 组数 {s['groups_total']} / 有产量 {s['groups_with_target']} / 样本 {s['samples_built']}"
        )
        if getattr(dataset, "static", None) is not None and "static_missing" in s:
            print(f"{name} 静态特征缺失样本 {s['static_missing']}")
        if s["days_expected"] > 0:
            avg_days = s["days_present"] / max(1, s["groups_total"])
            print(f"{name} 平均每天记录数 {avg_days:.1f} / 期望 {s['days_expected']}")
    x = dataset.x
    y = dataset.y
    print(
        f"{name} x 形状 {x.shape} y 形状 {y.shape} x均值 {x.mean():.4f} x标准差 {x.std():.4f} "
        f"y均值 {y.mean():.4f} y标准差 {y.std():.4f}"
    )


def train_one_epoch(model, loader, optimizer, device, use_static=False, criterion=None):
    """???? epoch????????"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        weekly = None
        if use_static:
            # ??????? + ????
            if len(batch) == 4:
                x, y, weekly, static = batch
            else:
                x, y, static = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            pred = model(x, static, weekly) if weekly is not None else model(x, static)
        else:
            # ????????
            if len(batch) == 3:
                x, y, weekly = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            pred = model(x, weekly) if weekly is not None else model(x)
        if criterion is None:
            loss = torch.mean((pred - y) ** 2)
        else:
            loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ????????????
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_one_epoch_graph(model, loader, optimizer, device, use_static=False, criterion=None):
    """???? epoch?????????????"""
    model.train()
    total_loss = 0.0
    total_nodes = 0.0
    province_loss_weight = float(getattr(model, "province_loss_weight", 1.0))
    enable_province_task = bool(getattr(model, "enable_province_task", False))
    for batch in loader:
        weekly = None
        province_y = None
        if use_static:
            if len(batch) == 7:
                x, y, weekly, static, mask, province_y, adj = batch
            else:
                x, y, static, mask, province_y, adj = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
        else:
            if len(batch) == 6:
                x, y, weekly, mask, province_y, adj = batch
            else:
                x, y, mask, province_y, adj = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            static = None
        mask = mask.to(device, non_blocking=True).float()
        adj = adj.to(device, non_blocking=True)
        if province_y is not None:
            province_y = province_y.to(device, non_blocking=True).view(-1)
        if use_static:
            pred = model(x, static, adj, mask, weekly) if weekly is not None else model(x, static, adj, mask)
        else:
            pred = model(x, adj, mask, weekly) if weekly is not None else model(x, adj, mask)
        if isinstance(pred, (tuple, list)):
            pred_county = pred[0]
            pred_province = pred[1] if len(pred) > 1 else None
        else:
            pred_county = pred
            pred_province = None
        if criterion is None:
            diff = (pred_county - y) * mask
            county_loss = (diff ** 2).sum() / mask.sum().clamp(min=1.0)
        else:
            county_loss = criterion(pred_county * mask, y * mask)
        loss = county_loss
        if enable_province_task and pred_province is not None and province_y is not None:
            if criterion is None:
                province_loss = torch.mean((pred_province - province_y) ** 2)
            else:
                province_loss = criterion(pred_province, province_y)
            loss = loss + province_loss_weight * province_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        total_nodes += mask.sum().item()
    return total_loss / max(1.0, total_nodes)


def eval_model(
    model,
    loader,
    device,
    y_mean=None,
    y_std=None,
    use_static=False,
    compute_loss=False,
    compute_metrics=False,
    criterion=None,
):
    """
    ???/?????????

    ??:
    - compute_loss: ???? MSE ??
    - compute_metrics: ???? RMSE/MAE/MAPE??? y_mean/y_std ?????
    """
    if compute_metrics and (y_mean is None or y_std is None):
        raise ValueError("????????? y_mean/y_std")
    model.eval()
    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    for batch in loader:
        weekly = None
        if use_static:
            if len(batch) == 4:
                x, y, weekly, static = batch
            else:
                x, y, static = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            pred = model(x, static, weekly) if weekly is not None else model(x, static)
        else:
            if len(batch) == 3:
                x, y, weekly = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            pred = model(x, weekly) if weekly is not None else model(x)
        batch_size = x.size(0)
        count += batch_size
        if compute_loss:
            if criterion is None:
                loss = torch.mean((pred - y) ** 2)
            else:
                loss = criterion(pred, y)
            total_loss += loss.item() * batch_size
        if compute_metrics:
            pred_raw = pred * y_std + y_mean
            y_raw = y * y_std + y_mean
            y_true.append(y_raw.detach().cpu())
            y_pred.append(pred_raw.detach().cpu())
    if compute_loss and not compute_metrics:
        return total_loss / max(1, count)
    if compute_metrics:
        if not y_true:
            metrics = (float("nan"), float("nan"), float("nan"))
        else:
            y_true_tensor = torch.cat(y_true, dim=0).view(-1)
            y_pred_tensor = torch.cat(y_pred, dim=0).view(-1)
            mse = torch.mean((y_pred_tensor - y_true_tensor) ** 2)
            rmse = float(torch.sqrt(mse).item())
            mae = float(torch.mean(torch.abs(y_pred_tensor - y_true_tensor)).item())
            denom = torch.abs(y_true_tensor).clamp(min=1e-6)
            mape = float((torch.mean(torch.abs(y_pred_tensor - y_true_tensor) / denom) * 100.0).item())
            metrics = (rmse, mae, mape)
        if compute_loss:
            return total_loss / max(1, count), metrics
        return metrics
    return None


def eval_model_graph(
    model,
    loader,
    device,
    y_mean=None,
    y_std=None,
    use_static=False,
    compute_loss=False,
    compute_metrics=False,
    criterion=None,
):
    """??????"""
    if compute_metrics and (y_mean is None or y_std is None):
        raise ValueError("????????? y_mean/y_std")
    model.eval()
    total_loss = 0.0
    total_nodes = 0.0
    y_true = []
    y_pred = []
    province_loss_weight = float(getattr(model, "province_loss_weight", 1.0))
    enable_province_task = bool(getattr(model, "enable_province_task", False))
    for batch in loader:
        weekly = None
        province_y = None
        if use_static:
            if len(batch) == 7:
                x, y, weekly, static, mask, province_y, adj = batch
            else:
                x, y, static, mask, province_y, adj = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
        else:
            if len(batch) == 6:
                x, y, weekly, mask, province_y, adj = batch
            else:
                x, y, mask, province_y, adj = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            static = None
        mask = mask.to(device, non_blocking=True).float()
        adj = adj.to(device, non_blocking=True)
        if province_y is not None:
            province_y = province_y.to(device, non_blocking=True).view(-1)
        if use_static:
            pred = model(x, static, adj, mask, weekly) if weekly is not None else model(x, static, adj, mask)
        else:
            pred = model(x, adj, mask, weekly) if weekly is not None else model(x, adj, mask)
        if isinstance(pred, (tuple, list)):
            pred_county = pred[0]
            pred_province = pred[1] if len(pred) > 1 else None
        else:
            pred_county = pred
            pred_province = None
        if compute_loss:
            province_loss = 0.0
            if enable_province_task and pred_province is not None and province_y is not None:
                if criterion is None:
                    province_loss = torch.mean((pred_province - province_y) ** 2)
                else:
                    province_loss = criterion(pred_province, province_y)
            if criterion is None:
                diff = (pred_county - y) * mask
                county_loss = (diff ** 2).sum() / mask.sum().clamp(min=1.0)
                total_loss += (county_loss + province_loss_weight * province_loss).item() * mask.sum().item()
                total_nodes += mask.sum().item()
            else:
                county_loss = criterion(pred_county * mask, y * mask)
                loss = county_loss + province_loss_weight * province_loss
                total_loss += loss.item() * mask.sum().item()
                total_nodes += mask.sum().item()
        if compute_metrics:
            pred_raw = pred_county * y_std + y_mean
            y_raw = y * y_std + y_mean
            mask_bool = mask.bool()
            y_true.append(y_raw[mask_bool].detach().cpu())
            y_pred.append(pred_raw[mask_bool].detach().cpu())
    if compute_loss and not compute_metrics:
        return total_loss / max(1.0, total_nodes)
    if compute_metrics:
        if not y_true:
            metrics = (float("nan"), float("nan"), float("nan"))
        else:
            y_true_tensor = torch.cat(y_true, dim=0).view(-1)
            y_pred_tensor = torch.cat(y_pred, dim=0).view(-1)
            mse = torch.mean((y_pred_tensor - y_true_tensor) ** 2)
            rmse = float(torch.sqrt(mse).item())
            mae = float(torch.mean(torch.abs(y_pred_tensor - y_true_tensor)).item())
            denom = torch.abs(y_true_tensor).clamp(min=1e-6)
            mape = float((torch.mean(torch.abs(y_pred_tensor - y_true_tensor) / denom) * 100.0).item())
            metrics = (rmse, mae, mape)
        if compute_loss:
            return total_loss / max(1.0, total_nodes), metrics
        return metrics
    return None


def collect_val_predictions(
    model,
    loader,
    device,
    y_mean,
    y_std,
    use_static=False,
    use_gnn=False,
):
    """Collect true/pred values in raw scale for scatter plots."""
    if y_mean is None or y_std is None:
        raise ValueError("y_mean/y_std required for prediction collection")
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        weekly = None
        if use_gnn:
            if use_static:
                if len(batch) == 7:
                    x, y, weekly, static, mask, _, adj = batch
                else:
                    x, y, static, mask, _, adj = batch
                static = static.to(device, non_blocking=True)
            else:
                if len(batch) == 6:
                    x, y, weekly, mask, _, adj = batch
                else:
                    x, y, mask, _, adj = batch
                static = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True).float()
            adj = adj.to(device, non_blocking=True)
            if use_static:
                pred = model(x, static, adj, mask, weekly) if weekly is not None else model(x, static, adj, mask)
            else:
                pred = model(x, adj, mask, weekly) if weekly is not None else model(x, adj, mask)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred_raw = pred * y_std + y_mean
            y_raw = y * y_std + y_mean
            mask_bool = mask.bool()
            y_true.append(y_raw[mask_bool].detach().cpu())
            y_pred.append(pred_raw[mask_bool].detach().cpu())
        else:
            if use_static:
                if len(batch) == 4:
                    x, y, weekly, static = batch
                else:
                    x, y, static = batch
                static = static.to(device, non_blocking=True)
            else:
                if len(batch) == 3:
                    x, y, weekly = batch
                else:
                    x, y = batch
                static = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if weekly is not None:
                weekly = weekly.to(device, non_blocking=True)
            pred = model(x, static, weekly) if use_static and weekly is not None else (
                model(x, static) if use_static else (model(x, weekly) if weekly is not None else model(x))
            )
            pred_raw = pred * y_std + y_mean
            y_raw = y * y_std + y_mean
            y_true.append(y_raw.detach().cpu())
            y_pred.append(pred_raw.detach().cpu())
    if not y_true:
        return np.array([]), np.array([])
    return (
        torch.cat(y_true, dim=0).view(-1).numpy(),
        torch.cat(y_pred, dim=0).view(-1).numpy(),
    )


def collect_graph_province_predictions(
    model,
    loader,
    device,
    y_mean,
    y_std,
    use_static=False,
):
    """Collect province-level true/pred values (raw scale) for graph dual-output mode."""
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        weekly = None
        province_y = None
        if use_static:
            if len(batch) == 7:
                x, _, weekly, static, mask, province_y, adj = batch
            else:
                x, _, static, mask, province_y, adj = batch
            static = static.to(device, non_blocking=True)
        else:
            if len(batch) == 6:
                x, _, weekly, mask, province_y, adj = batch
            else:
                x, _, mask, province_y, adj = batch
            static = None
        x = x.to(device, non_blocking=True)
        if weekly is not None:
            weekly = weekly.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).float()
        adj = adj.to(device, non_blocking=True)
        province_y = province_y.to(device, non_blocking=True).view(-1)
        if use_static:
            pred = model(x, static, adj, mask, weekly) if weekly is not None else model(x, static, adj, mask)
        else:
            pred = model(x, adj, mask, weekly) if weekly is not None else model(x, adj, mask)
        if not isinstance(pred, (tuple, list)) or len(pred) < 2:
            continue
        pred_province = pred[1]
        y_true_raw = province_y * y_std + y_mean
        y_pred_raw = pred_province * y_std + y_mean
        y_true.append(y_true_raw.detach().cpu())
        y_pred.append(y_pred_raw.detach().cpu())
    if not y_true:
        return np.array([]), np.array([])
    return (
        torch.cat(y_true, dim=0).view(-1).numpy(),
        torch.cat(y_pred, dim=0).view(-1).numpy(),
    )


def _sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    """在对数空间均匀采样，用于学习率等超参搜索。"""
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def _aggregate_rf_features(x: np.ndarray) -> np.ndarray:
    """把时序特征压缩成固定长度，供传统机器学习模型使用。"""
    # 用均值和标准差概括整条时间序列
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return np.concatenate([mean, std], axis=1)


class DailyLSTMBaseline(torch.nn.Module):
    """A small LSTM regressor over daily feature sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DailyTransformerBaseline(torch.nn.Module):
    """A small Transformer encoder regressor over daily feature sequences."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, model_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len + 1)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(model_dim),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.head(x[:, 0]).squeeze(-1)


class TemporalConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.residual = (
            torch.nn.Identity()
            if in_channels == out_channels
            else torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class GraphConvBlock(torch.nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,bcjt->bcit", adj, x)
        out = self.proj(agg)
        out = self.bn(out)
        out = torch.relu(out)
        return self.dropout(out)


class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.temporal1 = TemporalConvBlock(in_channels, out_channels, kernel_size, dropout)
        self.graph = GraphConvBlock(out_channels, dropout)
        self.temporal2 = TemporalConvBlock(out_channels, out_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.temporal1(x)
        x = self.graph(x, adj)
        x = self.temporal2(x)
        return x


class STGCNBaseline(torch.nn.Module):
    """A standard spatio-temporal graph baseline over county-year daily sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = input_dim
        for _ in range(num_blocks):
            blocks.append(STGCNBlock(in_channels, hidden_dim, kernel_size, dropout))
            in_channels = hidden_dim
        self.blocks = torch.nn.ModuleList(blocks)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, F] -> [B, F, N, T]
        x = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, adj)
        x = x.mean(dim=-1).permute(0, 2, 1)
        return self.head(x).squeeze(-1)


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = adj.astype(np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-6), -0.5)
    deg_inv_sqrt = np.diag(deg_inv_sqrt.astype(np.float32))
    return deg_inv_sqrt @ adj @ deg_inv_sqrt


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    return rmse, mae, mape


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size <= 1:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def append_county_prediction_rows(path, rows):
    """Append county-level rolling validation predictions."""
    if not rows:
        return
    fieldnames = ["model_name", "val_year", "county", "y_true", "y_pred", "error"]
    write_header = not os.path.exists(path)
    encoding = "utf-8-sig" if write_header else "utf-8"
    with open(path, "a", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_rf_baseline(
    args,
    x_all,
    y_all,
    years,
    fixed_seed,
):
    """
    训练并评估随机森林基线模型，用于和深度模型对比。
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except ImportError as exc:
        raise ImportError("RandomForest baseline requires scikit-learn") from exc

    val_years = list(range(args.roll_val_start, args.roll_val_end + 1))
    if not val_years:
        raise ValueError("Rolling validation years are empty; check roll-val-start/end")

    results_path = os.path.join(args.out_dir, "results.csv")
    result_fields = [
        "model_name",
        "val_year",
        "best_val_rmse",
        "best_val_mae",
        "best_val_mape",
        "best_epoch",
        "seed",
        "n_train_samples",
        "n_val_samples",
    ]

    val_metrics = []
    for val_year in val_years:
        train_mask = (years >= args.train_year_start) & (years <= val_year - 1)
        val_mask = years == val_year
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        x_train = _aggregate_rf_features(x_all[train_mask])
        y_train = y_all[train_mask]
        x_val = _aggregate_rf_features(x_all[val_mask])
        y_val = y_all[val_mask]

        best_params = None
        if args.rf_search:
            rng = random.Random(fixed_seed)
            best_rmse = None
            for trial in range(args.rf_trials):
                params = {
                    "n_estimators": rng.choice([100, 200, 400]),
                    "max_depth": rng.choice([5, 10, 15, None]),
                    "min_samples_leaf": rng.choice([1, 2, 5]),
                    "max_features": rng.choice(["sqrt", 0.7, 1.0]),
                }
                rf = RandomForestRegressor(
                    random_state=fixed_seed,
                    n_jobs=-1,
                    **params,
                )
                rf.fit(x_train, y_train)
                preds = rf.predict(x_val)
                rmse = float(math.sqrt(mean_squared_error(y_val, preds)))
                if best_rmse is None or rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
            if best_params is not None:
                print(f"Rolling year {val_year} RF best params {best_params}")

        if best_params is None:
            best_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
            }
        rf = RandomForestRegressor(
            random_state=fixed_seed,
            n_jobs=-1,
            **best_params,
        )
        rf.fit(x_train, y_train)
        preds = rf.predict(x_val)
        rmse = float(math.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        denom = np.maximum(np.abs(y_val), 1e-6)
        mape = float(np.mean(np.abs(preds - y_val) / denom) * 100.0)
        print(f"Val year {val_year} RF RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.2f}%")
        val_metrics.append((rmse, mae, mape))

        append_results_row(
            results_path,
            {
                "model_name": "rf",
                "val_year": int(val_year),
                "best_val_rmse": float(rmse),
                "best_val_mae": float(mae),
                "best_val_mape": float(mape),
                "best_epoch": None,
                "seed": int(fixed_seed),
                "n_train_samples": int(train_mask.sum()),
                "n_val_samples": int(val_mask.sum()),
            },
            result_fields,
        )
    if val_metrics:
        avg_rmse = sum(m[0] for m in val_metrics) / len(val_metrics)
        avg_mae = sum(m[1] for m in val_metrics) / len(val_metrics)
        avg_mape = sum(m[2] for m in val_metrics) / len(val_metrics)
        print(f"RF rolling avg RMSE {avg_rmse:.4f} MAE {avg_mae:.4f} MAPE {avg_mape:.2f}%")


def run_lstm_daily_baseline(
    args,
    x_all,
    y_all,
    years,
    counties,
    fixed_seed,
    device,
):
    """Train a simple daily-sequence LSTM baseline with rolling-year validation."""
    results_path = os.path.join(args.out_dir, "results.csv")
    preds_path = os.path.join(args.out_dir, "rollval_predictions.csv")
    for path in [results_path, preds_path]:
        if os.path.exists(path):
            os.remove(path)

    result_fields = [
        "model_name",
        "val_year",
        "best_val_rmse",
        "best_val_mae",
        "best_val_mape",
        "best_val_r2",
        "best_epoch",
        "seed",
        "n_train_samples",
        "n_val_samples",
    ]
    val_years = list(range(args.roll_val_start, args.roll_val_end + 1))
    if not val_years:
        raise ValueError("Rolling validation years are empty; check roll-val-start/end")

    val_metrics = []
    for val_year in val_years:
        set_seed(fixed_seed)
        train_mask = (years >= args.train_year_start) & (years <= val_year - 1)
        val_mask = years == val_year
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        x_mean, x_std = compute_feature_stats(x_all[train_mask])
        y_mean, y_std = compute_target_stats(y_all[train_mask])
        x_train = normalize_features(x_all[train_mask], x_mean, x_std).astype(np.float32)
        x_val = normalize_features(x_all[val_mask], x_mean, x_std).astype(np.float32)
        y_train = ((y_all[train_mask] - y_mean) / y_std).astype(np.float32)
        y_val = y_all[val_mask].astype(np.float32)
        val_counties = counties[val_mask]

        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train)
        x_val_t = torch.from_numpy(x_val).to(device)

        model = DailyLSTMBaseline(
            input_dim=x_train.shape[2],
            hidden_dim=args.daily_lstm_hidden,
            num_layers=args.daily_lstm_layers,
            dropout=args.daily_lstm_dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.daily_lstm_lr,
            weight_decay=args.weight_decay,
        )
        criterion = torch.nn.MSELoss()

        best_state = None
        best_epoch = 1
        best_rmse = None
        best_preds = None
        no_improve = 0
        n_train = x_train_t.shape[0]

        for epoch in range(1, args.daily_lstm_epochs + 1):
            model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, args.batch_size):
                idx = perm[start : start + args.batch_size]
                xb = x_train_t[idx].to(device)
                yb = y_train_t[idx].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val_norm = model(x_val_t).detach().cpu().numpy().astype(np.float32)
            pred_val = pred_val_norm * y_std + y_mean
            rmse, mae, mape = _compute_regression_metrics(y_val, pred_val)
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                best_preds = pred_val.copy()
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch == 1 or epoch % 20 == 0 or no_improve == 0:
                print(
                    f"Val year {val_year} LSTM-daily epoch {epoch:03d} "
                    f"RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.2f}%"
                )
            if no_improve >= args.daily_lstm_early_stop:
                break

        if best_state is None or best_preds is None or best_rmse is None:
            raise RuntimeError(f"LSTM-daily baseline failed for val year {val_year}")

        best_mae, best_mape = _compute_regression_metrics(y_val, best_preds)[1:]
        best_r2 = _compute_r2(y_val, best_preds)
        print(
            f"Val year {val_year} LSTM-daily best epoch {best_epoch} "
            f"RMSE {best_rmse:.4f} MAE {best_mae:.4f} MAPE {best_mape:.2f}% R2 {best_r2:.4f}"
        )
        val_metrics.append((best_rmse, best_mae, best_mape))

        append_results_row(
            results_path,
            {
                "model_name": "lstm_daily",
                "val_year": int(val_year),
                "best_val_rmse": float(best_rmse),
                "best_val_mae": float(best_mae),
                "best_val_mape": float(best_mape),
                "best_val_r2": float(best_r2),
                "best_epoch": int(best_epoch),
                "seed": int(fixed_seed),
                "n_train_samples": int(train_mask.sum()),
                "n_val_samples": int(val_mask.sum()),
            },
            result_fields,
        )
        pred_rows = []
        for county, y_true_i, y_pred_i in zip(val_counties.tolist(), y_val.tolist(), best_preds.tolist()):
            pred_rows.append(
                {
                    "model_name": "lstm_daily",
                    "val_year": int(val_year),
                    "county": str(county),
                    "y_true": float(y_true_i),
                    "y_pred": float(y_pred_i),
                    "error": float(y_pred_i - y_true_i),
                }
            )
        append_county_prediction_rows(preds_path, pred_rows)

    if val_metrics:
        avg_rmse = sum(m[0] for m in val_metrics) / len(val_metrics)
        avg_mae = sum(m[1] for m in val_metrics) / len(val_metrics)
        avg_mape = sum(m[2] for m in val_metrics) / len(val_metrics)
        print(
            f"LSTM-daily rolling avg RMSE {avg_rmse:.4f} "
            f"MAE {avg_mae:.4f} MAPE {avg_mape:.2f}%"
        )


def run_transformer_daily_baseline(
    args,
    x_all,
    y_all,
    years,
    counties,
    fixed_seed,
    device,
):
    """Train a simple attention-based daily Transformer baseline with rolling-year validation."""
    results_path = os.path.join(args.out_dir, "results.csv")
    preds_path = os.path.join(args.out_dir, "rollval_predictions.csv")
    for path in [results_path, preds_path]:
        if os.path.exists(path):
            os.remove(path)

    result_fields = [
        "model_name",
        "val_year",
        "best_val_rmse",
        "best_val_mae",
        "best_val_mape",
        "best_val_r2",
        "best_epoch",
        "seed",
        "n_train_samples",
        "n_val_samples",
    ]
    val_years = list(range(args.roll_val_start, args.roll_val_end + 1))
    if not val_years:
        raise ValueError("Rolling validation years are empty; check roll-val-start/end")

    val_metrics = []
    for val_year in val_years:
        set_seed(fixed_seed)
        train_mask = (years >= args.train_year_start) & (years <= val_year - 1)
        val_mask = years == val_year
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        x_mean, x_std = compute_feature_stats(x_all[train_mask])
        y_mean, y_std = compute_target_stats(y_all[train_mask])
        x_train = normalize_features(x_all[train_mask], x_mean, x_std).astype(np.float32)
        x_val = normalize_features(x_all[val_mask], x_mean, x_std).astype(np.float32)
        y_train = ((y_all[train_mask] - y_mean) / y_std).astype(np.float32)
        y_val = y_all[val_mask].astype(np.float32)
        val_counties = counties[val_mask]

        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train)
        x_val_t = torch.from_numpy(x_val).to(device)

        model = DailyTransformerBaseline(
            input_dim=x_train.shape[2],
            model_dim=args.daily_transformer_dim,
            num_heads=args.daily_transformer_heads,
            num_layers=args.daily_transformer_layers,
            ff_dim=args.daily_transformer_ff_dim,
            dropout=args.daily_transformer_dropout,
            max_len=x_train.shape[1],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.daily_transformer_lr,
            weight_decay=args.weight_decay,
        )
        criterion = torch.nn.MSELoss()

        best_epoch = 1
        best_rmse = None
        best_preds = None
        no_improve = 0
        n_train = x_train_t.shape[0]

        for epoch in range(1, args.daily_transformer_epochs + 1):
            model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, args.batch_size):
                idx = perm[start : start + args.batch_size]
                xb = x_train_t[idx].to(device)
                yb = y_train_t[idx].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val_norm = model(x_val_t).detach().cpu().numpy().astype(np.float32)
            pred_val = pred_val_norm * y_std + y_mean
            rmse, mae, mape = _compute_regression_metrics(y_val, pred_val)
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                best_preds = pred_val.copy()
                no_improve = 0
            else:
                no_improve += 1
            if epoch == 1 or epoch % 20 == 0 or no_improve == 0:
                print(
                    f"Val year {val_year} Transformer-daily epoch {epoch:03d} "
                    f"RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.2f}%"
                )
            if no_improve >= args.daily_transformer_early_stop:
                break

        if best_preds is None or best_rmse is None:
            raise RuntimeError(f"Transformer-daily baseline failed for val year {val_year}")

        best_mae, best_mape = _compute_regression_metrics(y_val, best_preds)[1:]
        best_r2 = _compute_r2(y_val, best_preds)
        print(
            f"Val year {val_year} Transformer-daily best epoch {best_epoch} "
            f"RMSE {best_rmse:.4f} MAE {best_mae:.4f} MAPE {best_mape:.2f}% R2 {best_r2:.4f}"
        )
        val_metrics.append((best_rmse, best_mae, best_mape))

        append_results_row(
            results_path,
            {
                "model_name": "transformer_daily",
                "val_year": int(val_year),
                "best_val_rmse": float(best_rmse),
                "best_val_mae": float(best_mae),
                "best_val_mape": float(best_mape),
                "best_val_r2": float(best_r2),
                "best_epoch": int(best_epoch),
                "seed": int(fixed_seed),
                "n_train_samples": int(train_mask.sum()),
                "n_val_samples": int(val_mask.sum()),
            },
            result_fields,
        )
        pred_rows = []
        for county, y_true_i, y_pred_i in zip(val_counties.tolist(), y_val.tolist(), best_preds.tolist()):
            pred_rows.append(
                {
                    "model_name": "transformer_daily",
                    "val_year": int(val_year),
                    "county": str(county),
                    "y_true": float(y_true_i),
                    "y_pred": float(y_pred_i),
                    "error": float(y_pred_i - y_true_i),
                }
            )
        append_county_prediction_rows(preds_path, pred_rows)

    if val_metrics:
        avg_rmse = sum(m[0] for m in val_metrics) / len(val_metrics)
        avg_mae = sum(m[1] for m in val_metrics) / len(val_metrics)
        avg_mape = sum(m[2] for m in val_metrics) / len(val_metrics)
        print(
            f"Transformer-daily rolling avg RMSE {avg_rmse:.4f} "
            f"MAE {avg_mae:.4f} MAPE {avg_mape:.2f}%"
        )


def run_stgcn_baseline(
    args,
    x_all,
    y_all,
    years,
    counties,
    fixed_seed,
    device,
):
    """Train a standard STGCN baseline with rolling-year validation."""
    full_county_order, full_adj = build_county_adjacency(args.geojson)
    if not full_county_order:
        raise ValueError(f"行政区划文件未解析到县名: {args.geojson}")
    data_counties = set(str(c) for c in counties.tolist())
    county_order = [c for c in full_county_order if c in data_counties]
    if not county_order:
        raise ValueError("GeoJSON 与数据中的县名没有交集，无法运行 STGCN 基线")
    idx = [full_county_order.index(c) for c in county_order]
    adj = full_adj[np.ix_(idx, idx)].astype(np.float32)
    adj_norm = torch.from_numpy(_normalize_adjacency(adj)).to(device)

    graph_samples, graph_years = build_graph_year_samples(
        x_all,
        y_all,
        years,
        counties,
        county_order,
        weekly=None,
        static=None,
        province_targets=None,
    )
    year_to_sample = {int(year): sample for year, sample in zip(graph_years, graph_samples)}

    results_path = os.path.join(args.out_dir, "results.csv")
    preds_path = os.path.join(args.out_dir, "rollval_predictions.csv")
    for path in [results_path, preds_path]:
        if os.path.exists(path):
            os.remove(path)

    result_fields = [
        "model_name",
        "val_year",
        "best_val_rmse",
        "best_val_mae",
        "best_val_mape",
        "best_val_r2",
        "best_epoch",
        "seed",
        "n_train_samples",
        "n_val_samples",
    ]
    val_metrics = []
    val_years = list(range(args.roll_val_start, args.roll_val_end + 1))
    if not val_years:
        raise ValueError("Rolling validation years are empty; check roll-val-start/end")

    for val_year in val_years:
        set_seed(fixed_seed)
        train_years = [y for y in graph_years if args.train_year_start <= int(y) <= val_year - 1]
        if not train_years or int(val_year) not in year_to_sample:
            continue
        train_samples = [year_to_sample[int(y)] for y in train_years]
        val_sample = year_to_sample[int(val_year)]

        x_train_flat = np.concatenate([s[0][s[4]] for s in train_samples], axis=0)
        y_train_flat = np.concatenate([s[1][s[4]] for s in train_samples], axis=0)
        x_mean, x_std = compute_feature_stats(x_train_flat)
        y_mean, y_std = compute_target_stats(y_train_flat)

        def _prep_sample(sample):
            x_year, y_year, _, _, mask, _ = sample
            x_norm = normalize_features(x_year, x_mean, x_std).astype(np.float32)
            y_norm = ((y_year - y_mean) / y_std).astype(np.float32)
            return (
                torch.from_numpy(x_norm).unsqueeze(0).to(device),
                torch.from_numpy(y_norm).unsqueeze(0).to(device),
                torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device),
            )

        train_tensors = [_prep_sample(sample) for sample in train_samples]
        x_val_t, y_val_t, mask_val_t = _prep_sample(val_sample)
        val_mask_np = val_sample[4].astype(bool)
        val_y_raw = val_sample[1][val_mask_np].astype(np.float32)
        val_counties = [county_order[i] for i, keep in enumerate(val_mask_np.tolist()) if keep]

        model = STGCNBaseline(
            input_dim=x_all.shape[2],
            hidden_dim=args.stgcn_hidden,
            kernel_size=args.stgcn_kernel_size,
            num_blocks=args.stgcn_blocks,
            dropout=args.stgcn_dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.stgcn_lr,
            weight_decay=args.weight_decay,
        )

        best_epoch = 1
        best_rmse = None
        best_preds = None
        no_improve = 0

        for epoch in range(1, args.stgcn_epochs + 1):
            model.train()
            order = list(range(len(train_tensors)))
            random.shuffle(order)
            for idx_train in order:
                xb, yb, mb = train_tensors[idx_train]
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb, adj_norm)
                loss = (((pred - yb) * mb) ** 2).sum() / mb.sum().clamp(min=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(x_val_t, adj_norm)
            pred_val_raw = (pred_val * y_std + y_mean).detach().cpu().numpy()[0]
            pred_val_raw = pred_val_raw[val_mask_np].astype(np.float32)
            rmse, mae, mape = _compute_regression_metrics(val_y_raw, pred_val_raw)
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                best_preds = pred_val_raw.copy()
                no_improve = 0
            else:
                no_improve += 1
            if epoch == 1 or epoch % 20 == 0 or no_improve == 0:
                print(
                    f"Val year {val_year} STGCN epoch {epoch:03d} "
                    f"RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.2f}%"
                )
            if no_improve >= args.stgcn_early_stop:
                break

        if best_preds is None or best_rmse is None:
            raise RuntimeError(f"STGCN baseline failed for val year {val_year}")

        best_mae, best_mape = _compute_regression_metrics(val_y_raw, best_preds)[1:]
        best_r2 = _compute_r2(val_y_raw, best_preds)
        print(
            f"Val year {val_year} STGCN best epoch {best_epoch} "
            f"RMSE {best_rmse:.4f} MAE {best_mae:.4f} MAPE {best_mape:.2f}% R2 {best_r2:.4f}"
        )
        val_metrics.append((best_rmse, best_mae, best_mape))

        append_results_row(
            results_path,
            {
                "model_name": "stgcn",
                "val_year": int(val_year),
                "best_val_rmse": float(best_rmse),
                "best_val_mae": float(best_mae),
                "best_val_mape": float(best_mape),
                "best_val_r2": float(best_r2),
                "best_epoch": int(best_epoch),
                "seed": int(fixed_seed),
                "n_train_samples": int(sum(int(sample[4].sum()) for sample in train_samples)),
                "n_val_samples": int(val_mask_np.sum()),
            },
            result_fields,
        )
        pred_rows = []
        for county, y_true_i, y_pred_i in zip(val_counties, val_y_raw.tolist(), best_preds.tolist()):
            pred_rows.append(
                {
                    "model_name": "stgcn",
                    "val_year": int(val_year),
                    "county": str(county),
                    "y_true": float(y_true_i),
                    "y_pred": float(y_pred_i),
                    "error": float(y_pred_i - y_true_i),
                }
            )
        append_county_prediction_rows(preds_path, pred_rows)

    if val_metrics:
        avg_rmse = sum(m[0] for m in val_metrics) / len(val_metrics)
        avg_mae = sum(m[1] for m in val_metrics) / len(val_metrics)
        avg_mape = sum(m[2] for m in val_metrics) / len(val_metrics)
        print(f"STGCN rolling avg RMSE {avg_rmse:.4f} MAE {avg_mae:.4f} MAPE {avg_mape:.2f}%")
def save_rollval_rmse_plot(rmse_histories, out_path, max_epoch: int = 80):
    """Rolling-year 验证 RMSE 收敛图（均值 ± 标准差）。"""
    if not rmse_histories:
        return
    max_len = min(max(len(h) for h in rmse_histories), max_epoch)
    if max_len <= 0:
        return
    rmse_arr = np.full((len(rmse_histories), max_len), np.nan, dtype=np.float32)
    for i, h in enumerate(rmse_histories):
        n = min(len(h), max_len)
        if n > 0:
            rmse_arr[i, :n] = np.array(h[:n], dtype=np.float32)
    epochs = np.arange(1, max_len + 1)
    mean = np.nanmean(rmse_arr, axis=0)
    std = np.nanstd(rmse_arr, axis=0)
    window = 5
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth = np.convolve(mean, kernel, mode="same")
    half = window // 2
    if smooth.size > 2 * half:
        smooth[:half] = np.nan
        smooth[-half:] = np.nan
    y_min = float(np.nanmin(mean - std))
    y_max = float(np.nanmax(mean + std))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    pad = (y_max - y_min) * 0.05
    if pad <= 0:
        pad = max(1e-3, abs(y_max) * 0.05)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, smooth, color="#E24A33", linewidth=2.5)
    plt.fill_between(epochs, mean - std, mean + std, color="#E24A33", alpha=0.11)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.xlim(1, max_len)
    plt.ylim(y_min - pad, y_max + pad)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_repr_rmse_plot(rmse_history, out_path, best_epoch, early_stop_epoch=None, max_epoch: int = 80):
    """代表年份验证 RMSE 收敛图（含 early-stop 与 best epoch 标注）。"""
    if not rmse_history:
        return
    max_len = min(len(rmse_history), max_epoch)
    if max_len <= 0:
        return
    epochs = np.arange(1, max_len + 1)
    rmse_arr = np.array(rmse_history[:max_len], dtype=np.float32)
    if not np.isfinite(rmse_arr).all():
        return
    window = 5
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth = np.convolve(rmse_arr, kernel, mode="same")
    half = window // 2
    if smooth.size > 2 * half:
        smooth[:half] = np.nan
        smooth[-half:] = np.nan
    p5 = float(np.percentile(rmse_arr, 5))
    p95 = float(np.percentile(rmse_arr, 95))
    margin = max(0.02, (p95 - p5) * 0.1)
    y_low = p5 - margin
    y_high = p95 + margin
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, rmse_arr, color="#AAAAAA", linewidth=1.0, alpha=0.25)
    plt.plot(epochs, smooth, color="#348ABD", linewidth=2.5)
    if early_stop_epoch is not None and early_stop_epoch <= max_len:
        plt.axvline(early_stop_epoch, color="#777777", linestyle="--", linewidth=1.5)
        plt.text(
            early_stop_epoch,
            y_high,
            "early stopping",
            color="#777777",
            fontsize=9,
            ha="center",
            va="bottom",
        )
    if best_epoch is not None and best_epoch <= max_len:
        plt.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.0)
        plt.text(
            best_epoch,
            y_low,
            "best epoch",
            fontsize=9,
            ha="left",
            va="bottom",
        )
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.xlim(1, max_len)
    plt.ylim(y_low, y_high)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_rollval_scatter_plot(year_points, out_path, max_points_per_year: int = 1500):
    """Rolling-year scatter plot of true vs predicted values."""
    if not year_points:
        return
    rng = np.random.default_rng(42)
    all_true = []
    all_pred = []
    all_years = []
    for year, (y_true, y_pred) in sorted(year_points.items()):
        if y_true.size == 0 or y_pred.size == 0:
            continue
        n = min(y_true.size, y_pred.size, max_points_per_year)
        if n <= 0:
            continue
        if y_true.size > n:
            idx = rng.choice(y_true.size, size=n, replace=False)
            y_true_s = y_true[idx]
            y_pred_s = y_pred[idx]
        else:
            y_true_s = y_true
            y_pred_s = y_pred
        all_true.append(y_true_s)
        all_pred.append(y_pred_s)
        all_years.append(np.full(y_true_s.shape, int(year), dtype=np.int32))
    if not all_true:
        return
    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    years_all = np.concatenate(all_years)
    lo = float(min(y_true_all.min(), y_pred_all.min()))
    hi = float(max(y_true_all.max(), y_pred_all.max()))
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        y_true_all,
        y_pred_all,
        s=14,
        alpha=0.4,
        marker="o",
        c=years_all,
        cmap="viridis",
        edgecolors="none",
    )
    plt.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("True yield")
    plt.ylabel("Predicted yield")
    cbar = plt.colorbar(sc)
    years_unique = sorted(set(int(y) for y in years_all.tolist()))
    if years_unique:
        cbar.set_ticks(years_unique)
    cbar.set_label("Year")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def append_results_row(path, row, fieldnames):
    """Append one row to results.csv, creating header if needed."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_province_prediction_rows(path, rows):
    """Append province-level rolling validation predictions."""
    if not rows:
        return
    fieldnames = ["model_name", "val_year", "y_true", "y_pred", "error"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def run_train_eval(
    args,
    train_loader,
    val_loader,
    stats,
    device,
    save_prefix: Optional[str] = None,
    use_static: bool = False,
    static_stats: Optional[Tuple] = None,
    use_gnn: bool = False,
):
    """
    训练模型并在验证集上早停，返回最佳模型及训练历史。
    """
    x_mean, x_std, y_mean, y_std = stats
    static_mean, static_std = (None, None)
    stage_lengths_list, stage_ratio_list = _get_stage_params(args)

    common_kwargs = {
        "feature_dim": getattr(args, "feature_dim", 9),
        "week_dim": args.week_dim,
        "transformer_heads": args.transformer_heads,
        "stage_hidden": args.stage_hidden,
        "season_hidden": args.season_hidden,
        "lstm_layers": args.lstm_layers,
        "stage_lengths": stage_lengths_list,
        "stage_ratio": stage_ratio_list,
        "dropout": args.dropout,
        "sif_week_idx": getattr(args, "sif_week_idx", None),
        "sif_missing_idx": getattr(args, "sif_missing_idx", None),
        "week_extra_dim": getattr(args, "week_extra_dim", 0),
    }

    if use_static:
        static_mean, static_std = static_stats if static_stats else (None, None)
        if use_gnn:
            model = DSTfusion(
                **common_kwargs,
                static_dim=getattr(args, "static_dim", 3),
                tower_output_dim=getattr(args, "tower_output_dim", 128),
                fusion_hidden=getattr(args, "fusion_hidden", 256),
                gnn_hidden=getattr(args, "gnn_hidden", 128),
                gnn_layers=getattr(args, "gnn_layers", 2),
                gnn_edge_dropout=getattr(args, "gnn_edge_dropout", 0.0),
                gnn_alpha0=getattr(args, "gnn_alpha0", 0.1),
                meteo_idx=getattr(args, "meteo_idx", []),
            ).to(device)
            model.enable_province_task = bool(getattr(args, "enable_province_task", False))
            model.province_loss_weight = (
                float(getattr(args, "province_loss_weight", 1.0))
                if model.enable_province_task
                else 0.0
            )
        else:
            model = DualTowerDDCN(
                **common_kwargs,
                static_dim=getattr(args, "static_dim", 3),
                tower_output_dim=getattr(args, "tower_output_dim", 128),
                fusion_hidden=getattr(args, "fusion_hidden", 256),
                meteo_idx=getattr(args, "meteo_idx", []),
            ).to(device)
    else:
        # 单塔 DDCN
        if use_gnn:
            raise ValueError("图模型需要 --use-dual-tower")
        model = DDCN(**common_kwargs).to(device)

    model_args = get_model_args(args, use_static)
    stats_with_static = (x_mean, x_std, y_mean, y_std, static_mean, static_std)
    # Adam 优化器 + L2 正则（排除 gnn_alpha）
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("gnn_alpha"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    lr = 5e-4 if not use_static else args.lr
    optimizer = torch.optim.Adam(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    if use_static:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            cooldown=2,
            threshold=1e-4,
            min_lr=1e-6,
        )
    else:
        scheduler = None

    best_val_loss = None
    best_r2 = None
    best_state = None
    best_epoch = None
    best_metrics = None
    history = {"train": [], "val": [], "val_rmse": [], "lr": []}
    no_improve = 0
    criterion = torch.nn.MSELoss() if not use_static else torch.nn.HuberLoss()
    early_stop_epoch = None
    for epoch in range(1, args.epochs + 1):
        # 训练一个 epoch
        if use_gnn:
            train_loss = train_one_epoch_graph(
                model,
                train_loader,
                optimizer,
                device,
                use_static=use_static,
                criterion=criterion,
            )
        else:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                use_static=use_static,
                criterion=criterion,
            )
        history["train"].append(train_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if val_loader is not None:
            if use_gnn:
                val_loss, metrics = eval_model_graph(
                    model,
                    val_loader,
                    device,
                    y_mean,
                    y_std,
                    use_static=use_static,
                    compute_loss=True,
                    compute_metrics=True,
                    criterion=criterion,
                )
            else:
                val_loss, metrics = eval_model(
                    model,
                    val_loader,
                    device,
                    y_mean,
                    y_std,
                    use_static=use_static,
                    compute_loss=True,
                    compute_metrics=True,
                    criterion=criterion,
                )
            val_rmse, val_mae, val_mape = metrics
            history["val"].append(val_loss)
            history["val_rmse"].append(val_rmse)
            score = val_rmse if not use_static else val_loss
            if best_val_loss is None or score < best_val_loss - 1e-4:
                best_val_loss = score
                best_r2 = None
                best_metrics = metrics
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
                if save_prefix:
                    save_checkpoint(
                        f"{save_prefix}_best.pt",
                        best_state,
                        stats_with_static,
                        use_static,
                        model_args,
                        extra={"val_rmse": val_rmse, "val_mae": val_mae, "val_mape": val_mape},
                    )
            else:
                no_improve += 1
            print(
                f"第{epoch:03d}轮 训练损失 {train_loss:.4f} 验证损失 {val_loss:.4f} "
                f"RMSE {val_rmse:.4f} MAE {val_mae:.4f} MAPE {val_mape:.2f}%"
            )
            if scheduler is not None:
                scheduler.step(score)
            patience = 20 if not use_static else args.early_stop
            if no_improve >= patience:
                print(f"验证集连续 {patience} 轮未提升，提前停止")
                early_stop_epoch = epoch
                break
        else:
            print(f"第{epoch:03d}轮 训练损失 {train_loss:.4f}")

    if best_state is not None:
        # 还原最佳轮次的权重
        model.load_state_dict(best_state, strict=True)
    if best_epoch is None:
        best_epoch = args.epochs
    # 如有验证集，额外计算 RMSE 便于排序/展示
    val_rmse = None
    if val_loader is not None and best_metrics is not None:
        val_rmse = best_metrics[0]
    return model, best_val_loss, val_rmse, history, best_epoch, best_metrics, early_stop_epoch


def main():
    """
    训练入口：
    - 解析参数
    - 读取数据并按年份划分
    - 训练/评估/保存模型或基线
    """
    # 调试模式下的默认参数，避免手动输入命令行
    debug_defaults = {
        "train": "data/All_Data.csv",
        "val": None,
        "feature_cols": "WDRVI_median,GCI_median,EVI_median,NDWI_median,NIRv_median,GDD,KDD,PRCP,VPD",
        "yield_csv": "data/Yield_Data.xlsx",
        "yield_col": "yield",
        "yield_county_col": "name",
        "yield_year_col": "year",
        "date_col": "Date",
        "county_col": "County",
    }

    # 命令行参数定义
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["rf", "single", "dual", "lstm_daily", "transformer_daily", "stgcn"],
        default="dual",
        help="训练模式：rf 随机森林 / single 单流 / dual 双塔 / lstm_daily 日尺度LSTM基线 / transformer_daily 注意力时序基线 / stgcn 标准时空图基线",
    )
    parser.add_argument("--train", required=False, help="训练集路径（仅支持 .csv）")
    parser.add_argument("--val", default=None, help="验证集路径（仅支持 .csv，可选）")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 工作进程数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--stage-lengths", type=str, default="5,8,3,3,6", help="阶段周数，如 5,5,5,5,5")
    parser.add_argument("--stage-ratio", type=str, default=None, help="阶段比例，如 10,11")
    parser.add_argument("--week-dim", type=int, default=128, help="周特征维度")
    parser.add_argument("--transformer-heads", type=int, default=2, help="Transformer 注意力头数")
    parser.add_argument("--stage-hidden", type=int, default=256, help="阶段 LSTM 隐藏维度")
    parser.add_argument("--season-hidden", type=int, default=256, help="季节 LSTM 隐藏维度")
    parser.add_argument("--lstm-layers", type=int, default=1, help="LSTM 层数")
    parser.add_argument("--model-out", default="model_full.pt", help="模型保存路径（可为相对路径）")
    parser.add_argument("--feature-cols", default=None, help="CSV 的特征列名，逗号分隔")
    parser.add_argument("--target-col", default="yield", help="日数据中的产量列名")
    parser.add_argument("--date-col", default="Date", help="日期列名")
    parser.add_argument("--county-col", default="County", help="县/区域列名")
    parser.add_argument("--yield-csv", default=None, help="独立产量文件（.csv 或 .xlsx）")
    parser.add_argument("--yield-col", default="yield", help="产量列名")
    parser.add_argument("--yield-county-col", default="name", help="产量文件中的县/区域列名")
    parser.add_argument("--yield-year-col", default="year", help="产量文件中的年份列名")
    parser.add_argument("--province-yield-csv", default="data/吉林省单产2000-2023.xlsx", help="省级单产文件（.csv 或 .xlsx）")
    parser.add_argument("--province-yield-col", default="单产（公斤/公顷）", help="省级单产列名")
    parser.add_argument("--province-yield-year-col", default="Unnamed: 0", help="省级年份列名")
    parser.add_argument("--province-yield-divisor", type=float, default=1000.0, help="省级单产缩放除数")
    parser.add_argument("--weekly-csv", default="data/SIF_Weekly.csv", help="Weekly feature CSV path")
    parser.add_argument(
        "--weekly-cols",
        default="SIF_median,sif_missing_flag",
        help="Weekly feature columns (comma-separated)",
    )
    parser.add_argument("--weekly-county-col", default="name", help="County column in weekly CSV")
    parser.add_argument("--weekly-year-col", default="year", help="Year column in weekly CSV")
    parser.add_argument(
        "--weekly-start-doy-col", default="week_start_doy", help="Weekly start DOY column"
    )
    parser.add_argument(
        "--weekly-end-doy-col", default="week_end_doy", help="Weekly end DOY column"
    )
    parser.add_argument("--start-doy", type=int, default=128, help="开始积日（DOY）")
    parser.add_argument("--end-doy", type=int, default=302, help="结束积日（DOY）")
    parser.add_argument("--fill-missing", choices=["ffill", "zero"], default="ffill", help="缺失值填充方式")
    parser.add_argument("--train-year-start", type=int, default=2001, help="训练集起始年")
    parser.add_argument("--train-year-end", type=int, default=2014, help="训练集截止年")
    parser.add_argument("--roll-val-start", type=int, default=2015, help="滚动训练开始年")
    parser.add_argument("--roll-val-end", type=int, default=2019, help="滚动训练截止年")
    parser.add_argument("--random-search", action="store_true", help="启用随机搜索超参数")
    parser.add_argument("--trials", type=int, default=10, help="随机搜索试验次数")
    # Seed is fixed to ensure reproducibility across runs.
    parser.add_argument("--rf-baseline", action="store_true", help="使用随机森林基线模型（等同于 --mode rf）")
    parser.add_argument(
        "--lstm-daily-baseline",
        action="store_true",
        help="使用日尺度 LSTM 基线模型（等同于 --mode lstm_daily）",
    )
    parser.add_argument(
        "--transformer-daily-baseline",
        action="store_true",
        help="使用日尺度 Transformer 基线模型（等同于 --mode transformer_daily）",
    )
    parser.add_argument(
        "--stgcn-baseline",
        action="store_true",
        help="使用标准 STGCN 基线模型（等同于 --mode stgcn）",
    )
    parser.add_argument("--rf-search", action="store_true", help="随机森林超参数搜索")
    parser.add_argument("--rf-trials", type=int, default=20, help="随机森林搜索试验次数")
    parser.add_argument("--daily-lstm-hidden", type=int, default=128, help="LSTM-daily 隐藏维度")
    parser.add_argument("--daily-lstm-layers", type=int, default=1, help="LSTM-daily 层数")
    parser.add_argument("--daily-lstm-dropout", type=float, default=0.2, help="LSTM-daily dropout")
    parser.add_argument("--daily-lstm-lr", type=float, default=1e-3, help="LSTM-daily 学习率")
    parser.add_argument("--daily-lstm-epochs", type=int, default=300, help="LSTM-daily 最大训练轮数")
    parser.add_argument("--daily-lstm-early-stop", type=int, default=30, help="LSTM-daily 早停轮数")
    parser.add_argument("--daily-transformer-dim", type=int, default=128, help="Transformer-daily 隐层维度")
    parser.add_argument("--daily-transformer-heads", type=int, default=4, help="Transformer-daily 头数")
    parser.add_argument("--daily-transformer-layers", type=int, default=2, help="Transformer-daily 层数")
    parser.add_argument("--daily-transformer-ff-dim", type=int, default=256, help="Transformer-daily 前馈维度")
    parser.add_argument("--daily-transformer-dropout", type=float, default=0.2, help="Transformer-daily dropout")
    parser.add_argument("--daily-transformer-lr", type=float, default=1e-3, help="Transformer-daily 学习率")
    parser.add_argument("--daily-transformer-epochs", type=int, default=300, help="Transformer-daily 最大训练轮数")
    parser.add_argument("--daily-transformer-early-stop", type=int, default=30, help="Transformer-daily 早停轮数")
    parser.add_argument("--stgcn-hidden", type=int, default=64, help="STGCN 隐层通道数")
    parser.add_argument("--stgcn-blocks", type=int, default=2, help="STGCN block 数")
    parser.add_argument("--stgcn-kernel-size", type=int, default=3, help="STGCN 时间卷积核大小")
    parser.add_argument("--stgcn-dropout", type=float, default=0.2, help="STGCN dropout")
    parser.add_argument("--stgcn-lr", type=float, default=1e-3, help="STGCN 学习率")
    parser.add_argument("--stgcn-epochs", type=int, default=300, help="STGCN 最大训练轮数")
    parser.add_argument("--stgcn-early-stop", type=int, default=30, help="STGCN 早停轮数")
    parser.add_argument("--dropout", type=float, default=0.2112762832922651, help="丢弃率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减（L2 正则）")
    parser.add_argument("--early-stop", type=int, default=30, help="验证集早停轮数")
    parser.add_argument("--out-dir", default="outputs", help="输出目录（模型与图表）")
    # 双塔模型参数
    parser.add_argument(
        "--use-dual-tower",
        action="store_true",
        default=True,
        help="使用双塔模型（动态+静态），默认开启",
    )
    parser.add_argument(
        "--no-dual-tower",
        dest="use_dual_tower",
        action="store_false",
        help="禁用双塔模型（等同于 --mode single）",
    )
    parser.add_argument("--static-csv", default="data/jilin_phenology_2001-2020.csv", help="静态特征数据文件（.csv）")
    parser.add_argument("--soil-csv", default="data/Jilin_Soil_Texture_Combined_Static.csv", help="土壤静态特征文件（.csv）")
    parser.add_argument(
        "--static-cols",
        default="SOS,VGS,RGS,sand_median,silt_median,clay_median",
        help="静态特征列名，逗号分隔",
    )
    parser.add_argument(
        "--soil-cols",
        default="sand_median,silt_median,clay_median",
        help="土壤静态列名（来自 soil-csv），逗号分隔",
    )
    parser.add_argument(
        "--meteo-cols",
        default="GDD,KDD,PRCP,VPD",
        help="用于气象通道的动态特征列名，逗号分隔",
    )
    parser.add_argument(
        "--sif-week-col",
        default="SIF_median",
        help="动态特征中的周 SIF 列名（用于 stage FiLM 调制）",
    )
    parser.add_argument(
        "--sif-missing-col",
        default="sif_missing_flag",
        help="动态特征中的 SIF 缺失标记列名（0/1）",
    )
    parser.add_argument("--tower-output-dim", type=int, default=128, help="双塔模型中每个塔的输出维度")
    parser.add_argument("--fusion-hidden", type=int, default=256, help="双塔融合层隐藏维度")
    parser.add_argument("--use-gnn", action="store_true", default=True, help="在季节特征后使用图卷积")
    parser.add_argument(
        "--no-gnn",
        dest="use_gnn",
        action="store_false",
        help="禁用图卷积",
    )
    parser.add_argument("--geojson", default="data/吉林省.json", help="行政区划 GeoJSON 路径")
    parser.add_argument("--gnn-hidden", type=int, default=128, help="图卷积隐藏维度")
    parser.add_argument("--gnn-layers", type=int, default=1, help="图卷积层数")
    parser.add_argument("--gnn-alpha0", type=float, default=0.005, help="图平滑残差初始强度(0-1)")
    parser.add_argument("--gnn-edge-dropout", type=float, default=0, help="图边随机丢弃率")
    parser.add_argument("--enable-province-task", action="store_true", help="启用省级联合训练与验证输出")
    parser.add_argument("--province-loss-weight", type=float, default=1.0, help="图模型省级损失权重")
    args = parser.parse_args()

    argv_flags = set(sys.argv[1:])
    if "--mode" not in argv_flags:
        if "--rf-baseline" in argv_flags:
            args.mode = "rf"
        elif "--lstm-daily-baseline" in argv_flags:
            args.mode = "lstm_daily"
        elif "--transformer-daily-baseline" in argv_flags:
            args.mode = "transformer_daily"
        elif "--stgcn-baseline" in argv_flags:
            args.mode = "stgcn"
        elif "--no-dual-tower" in argv_flags:
            args.mode = "single"

    if args.mode == "rf":
        args.rf_baseline = True
        args.lstm_daily_baseline = False
        args.transformer_daily_baseline = False
        args.stgcn_baseline = False
        args.use_dual_tower = False
        args.use_gnn = False
    elif args.mode == "lstm_daily":
        args.rf_baseline = False
        args.lstm_daily_baseline = True
        args.transformer_daily_baseline = False
        args.stgcn_baseline = False
        args.use_dual_tower = False
        args.use_gnn = False
        if "--feature-cols" not in argv_flags:
            args.feature_cols = (
                "WDRVI_median,GCI_median,EVI_median,NDWI_median,"
                "NIRv_median,GDD,KDD,PRCP,VPD"
            )
    elif args.mode == "transformer_daily":
        args.rf_baseline = False
        args.lstm_daily_baseline = False
        args.transformer_daily_baseline = True
        args.stgcn_baseline = False
        args.use_dual_tower = False
        args.use_gnn = False
        if "--feature-cols" not in argv_flags:
            args.feature_cols = (
                "WDRVI_median,GCI_median,EVI_median,NDWI_median,"
                "NIRv_median,GDD,KDD,PRCP,VPD"
            )
    elif args.mode == "stgcn":
        args.rf_baseline = False
        args.lstm_daily_baseline = False
        args.transformer_daily_baseline = False
        args.stgcn_baseline = True
        args.use_dual_tower = False
        args.use_gnn = False
        if "--feature-cols" not in argv_flags:
            args.feature_cols = (
                "WDRVI_median,GCI_median,EVI_median,NDWI_median,"
                "NIRv_median,GDD,KDD,PRCP,VPD"
            )
    elif args.mode == "single":
        args.rf_baseline = False
        args.lstm_daily_baseline = False
        args.transformer_daily_baseline = False
        args.stgcn_baseline = False
        args.use_dual_tower = False
        args.use_gnn = False
        if "--lr" not in argv_flags:
            args.lr = 1e-4
        if "--stage-lengths" not in argv_flags:
            args.stage_lengths = "4,5,4,4,4"
        if "--end-doy" not in argv_flags:
            args.end_doy = 274
        if "--feature-cols" not in argv_flags:
            args.feature_cols = (
                "WDRVI_median,GCI_median,EVI_median,NDWI_median,"
                "NIRv_median,GDD,KDD,PRCP,VPD"
            )
    else:
        args.rf_baseline = False
        args.lstm_daily_baseline = False
        args.transformer_daily_baseline = False
        args.stgcn_baseline = False
        args.use_dual_tower = True
        if "--feature-cols" not in argv_flags:
            args.feature_cols = (
                "WDRVI_median,GCI_median,EVI_median,NDWI_median,NIRv_median,"
                "GDD,KDD,PRCP,VPD,"
                "WDRVI_p90,WDRVI_p10,GCI_p90,GCI_p10,EVI_p90,EVI_p10,"
                "NDWI_p90,NDWI_p10,NIRv_p90,NIRv_p10"
            )

    # 固定随机种子，方便复现实验
    fixed_seed = 42
    set_seed(fixed_seed)

    # 如果没有命令行参数或未指定训练集，则自动填入调试默认值
    if len(sys.argv) == 1 or not args.train:
        args.train = debug_defaults["train"]
        args.val = debug_defaults["val"]
        if not args.feature_cols:
            args.feature_cols = debug_defaults["feature_cols"]
        args.yield_csv = debug_defaults["yield_csv"]
        args.yield_col = debug_defaults["yield_col"]
        args.yield_county_col = debug_defaults["yield_county_col"]
        args.yield_year_col = debug_defaults["yield_year_col"]
        args.date_col = debug_defaults["date_col"]
        args.county_col = debug_defaults["county_col"]

    if not args.train:
        raise ValueError("训练集路径不能为空，请设置 --train")

    # 选择计算设备：有 GPU 就用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "rf":
        mode_dir = "rf_baseline"
    elif args.mode == "lstm_daily":
        mode_dir = "lstm_daily_baseline"
    elif args.mode == "transformer_daily":
        mode_dir = "transformer_daily_baseline"
    elif args.mode == "stgcn":
        mode_dir = "stgcn_baseline"
    elif args.mode == "single":
        mode_dir = "single_tower"
    else:
        mode_dir = "dual_tower"
    args.out_dir = os.path.join(args.out_dir, mode_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(
        "启动模式: {mode} | dual_tower={dual} gnn={gnn} rf={rf} | "
        "stage_lengths={stage_lengths} end_doy={end_doy} | feature_cols={feature_cols}".format(
            mode=args.mode,
            dual=args.use_dual_tower,
            gnn=getattr(args, "use_gnn", False),
            rf=args.rf_baseline,
            stage_lengths=args.stage_lengths,
            end_doy=args.end_doy,
            feature_cols=args.feature_cols,
        )
    )

    if not args.train.lower().endswith(".csv"):
        raise ValueError("仅支持按年划分，请使用 .csv 训练集文件。")
    if args.train.lower().endswith(".csv"):
        # 读取全量 CSV，并按年份范围划分训练/验证/测试
        cols = [c.strip() for c in (args.feature_cols or "").split(",") if c.strip()]
        static_cols = None
        static_cols_expanded = None
        if args.use_dual_tower and args.static_cols:
            static_cols = [c.strip() for c in args.static_cols.split(",") if c.strip()]
            static_cols_expanded = expand_static_cols(static_cols)
            static_cols_expanded = static_cols_expanded + [
                "hist_yield_mean",
                "hist_yield_missing_flag",
            ]

        meteo_cols = [c.strip() for c in (args.meteo_cols or "").split(",") if c.strip()]
        args.meteo_idx = _resolve_col_indices(cols, meteo_cols, "meteo")

        weekly_cols = [c.strip() for c in (args.weekly_cols or "").split(",") if c.strip()]
        args.week_extra_dim = len(weekly_cols)
        args.sif_week_idx = None
        args.sif_missing_idx = None
        if weekly_cols:
            if args.sif_week_col:
                args.sif_week_idx = _resolve_first_existing_index(
                    weekly_cols,
                    [args.sif_week_col, "SIF_median", "sif_median"],
                    "sif_week",
                    required=True,
                )
            if args.sif_missing_col:
                args.sif_missing_idx = _resolve_first_existing_index(
                    weekly_cols,
                    [args.sif_missing_col, "sif_missing", "sif_missing_flag"],
                    "sif_missing",
                    required=True,
                )

        # 根据列名自动推导特征维度
        args.feature_dim = len(cols)
        if args.use_dual_tower:
            args.static_dim = len(static_cols_expanded or [])
        
        if args.use_dual_tower and not static_cols:
            raise ValueError("双塔模型需要 --static-cols")
        full_ds = CsvYieldDataset(
            args.train,
            feature_cols=cols,
            target_col=args.target_col,
            date_col=args.date_col,
            county_col=args.county_col,
            yield_csv=args.yield_csv,
            yield_col=args.yield_col,
            yield_county_col=args.yield_county_col,
            yield_year_col=args.yield_year_col,
            weekly_csv=args.weekly_csv,
            weekly_cols=[c.strip() for c in (args.weekly_cols or "").split(",") if c.strip()],
            weekly_county_col=args.weekly_county_col,
            weekly_year_col=args.weekly_year_col,
            weekly_start_doy_col=args.weekly_start_doy_col,
            weekly_end_doy_col=args.weekly_end_doy_col,
            start_doy=args.start_doy,
            end_doy=args.end_doy,
            fill_missing=args.fill_missing,
            static_cols=static_cols,
            static_csv=args.static_csv if args.use_dual_tower else None,
            soil_csv=args.soil_csv if args.use_dual_tower else None,
            soil_cols=[c.strip() for c in (args.soil_cols or "").split(",") if c.strip()],
        )

        # 提取数据数组和年份信息，便于后续按年切分
        years = full_ds.years
        x_all = full_ds.x
        y_all = full_ds.y
        weekly_all = full_ds.weekly
        static_all = full_ds.static if args.use_dual_tower else None
        static_mask_all = full_ds.static_mask if args.use_dual_tower else None
        counties = full_ds.counties

        if args.use_dual_tower and static_all is None:
            raise ValueError("静态特征缺失，请检查 --static-csv/--static-cols")

        print_dataset_stats("全部数据", full_ds)

        graph_samples_cache = None
        graph_sample_years = None
        province_targets_by_year = None
        if args.use_gnn:
            if not args.use_dual_tower:
                raise ValueError("图模型需要 --use-dual-tower")
            if args.enable_province_task:
                if not args.province_yield_csv:
                    candidates = []
                    for pattern in ("*.xlsx", "*.xls", "*.csv"):
                        candidates.extend(glob.glob(pattern))
                    candidates = [
                        p for p in candidates
                        if not os.path.basename(p).startswith("~$")
                        and os.path.basename(p).lower() not in {"yield_data.xlsx", "all_data.csv"}
                    ]
                    if not candidates:
                        raise ValueError("使用图模型双输出时需要提供 --province-yield-csv")
                    args.province_yield_csv = sorted(candidates)[0]
                    print(f"未指定 --province-yield-csv，自动使用: {args.province_yield_csv}")
                province_targets_by_year = load_province_year_targets(
                    args.province_yield_csv,
                    year_col=args.province_yield_year_col or None,
                    target_col=args.province_yield_col or None,
                    target_divisor=args.province_yield_divisor,
                )
                if not province_targets_by_year:
                    raise ValueError("未读取到省级单产标签，请检查 --province-yield-csv/列名")
            county_order, adj = build_county_adjacency(args.geojson)
            if not county_order:
                raise ValueError(f"行政区划文件未解析到县名: {args.geojson}")
            data_counties = set(str(c) for c in counties.tolist())
            county_order_s = [c for c in county_order if c in data_counties]
            if not county_order_s:
                raise ValueError("行政区划与数据县名无交集，请检查县名字段")
            missing_in_geo = sorted(data_counties - set(county_order_s))
            if missing_in_geo:
                print(f"警告: {len(missing_in_geo)} 个县未在行政区划中找到，将被忽略")
            idx = [county_order.index(c) for c in county_order_s]
            adj_s = adj[np.ix_(idx, idx)]
            args.graph_county_order = county_order_s
            args.graph_adj = adj_s
            graph_samples_cache, graph_sample_years = build_graph_year_samples(
                x_all,
                y_all,
                years,
                counties,
                args.graph_county_order,
                weekly_all,
                static_all,
                province_targets=province_targets_by_year,
            )

        # 训练年份范围
        train_mask = (years >= args.train_year_start) & (years <= args.train_year_end)
        if train_mask.sum() == 0:
            raise ValueError("训练划分为空，请检查年份范围")

        # 滚动验证年份列表（可选）
        val_years = list(range(args.roll_val_start, args.roll_val_end + 1))
        if not val_years:
            raise ValueError("滚动验证年份为空，请检查 roll-val-start/end")
        if args.random_search and not val_years:
            raise ValueError("随机搜索依赖滚动验证，请检查 roll-val-start/end")

        if args.rf_baseline:
            # 如果选择基线模型，直接运行随机森林并退出
            run_rf_baseline(
                args,
                x_all,
                y_all,
                years,
                fixed_seed,
            )
            return
        if args.lstm_daily_baseline:
            run_lstm_daily_baseline(
                args,
                x_all,
                y_all,
                years,
                counties,
                fixed_seed,
                device,
            )
            return
        if args.transformer_daily_baseline:
            run_transformer_daily_baseline(
                args,
                x_all,
                y_all,
                years,
                counties,
                fixed_seed,
                device,
            )
            return
        if args.stgcn_baseline:
            run_stgcn_baseline(
                args,
                x_all,
                y_all,
                years,
                counties,
                fixed_seed,
                device,
            )
            return

        # 仅用训练集计算标准化统计量
        stats_full = _compute_stats_for_mask(
            x_all,
            y_all,
            static_all,
            static_mask_all,
            train_mask,
            args.use_dual_tower,
        )
        x_mean_full, x_std_full, y_mean_full, y_std_full, static_mean_full, static_std_full = stats_full
        
        print(
            f"固定划分 训练 {args.train_year_start}-{args.train_year_end}，"
            f"验证 {args.roll_val_start}-{args.roll_val_end}，"
            f"训练样本 {train_mask.sum()}"
        )
        if args.use_dual_tower:
            print(f"使用双塔模型，静态特征列: {static_cols_expanded or static_cols}")
            if static_mean_full is not None:
                print(f"静态特征统计: mean {static_mean_full}, std {static_std_full}")

    def build_loader(mask, batch_size, shuffle, stats):
        if args.use_gnn:
            samples = None
            sample_years = None
            if graph_samples_cache is not None and graph_sample_years is not None:
                year_set = set(int(y) for y in years[mask])
                samples = [s for s, yr in zip(graph_samples_cache, graph_sample_years) if yr in year_set]
                sample_years = [yr for yr in graph_sample_years if yr in year_set]
            return _build_graph_loader_for_mask(
                x_all,
                y_all,
                weekly_all,
                years,
                counties,
                mask,
                batch_size,
                shuffle,
                args.num_workers,
                stats,
                static_all,
                args.graph_county_order,
                args.graph_adj,
                province_targets=province_targets_by_year,
                samples=samples,
                sample_years=sample_years,
            )
        return _build_array_loader_for_mask(
            x_all,
            y_all,
            weekly_all,
            years,
            mask,
            batch_size,
            shuffle,
            args.num_workers,
            stats,
            static_all,
            static_mask_all,
            args.use_dual_tower,
        )

    if args.random_search:
        # 随机搜索超参数，挑选验证集表现最佳的一组
        rng = random.Random(fixed_seed)
        best_state = None
        best_history = None
        best_val_metrics = None
        best_params = None
        best_epoch = None
        best_stats = None
        best_val_rmse = None

        for trial in range(args.trials):
            trial_args = deepcopy(args)
            trial_args.lr = _sample_log_uniform(rng, 1e-4, 5e-3)
            trial_args.batch_size = rng.choice([16, 32, 64])
            trial_args.dropout = rng.uniform(0.1, 0.5)
            trial_args.transformer_heads = rng.choice([2, 4, 8])
            trial_args.lstm_layers = rng.choice([1, 2])
            print(
                f"随机搜索试验 {trial + 1}/{args.trials}: 学习率 {trial_args.lr:.6f} "
                f"批大小 {trial_args.batch_size} 丢弃率 {trial_args.dropout:.2f} "
                f"注意力头数 {trial_args.transformer_heads} LSTM层数 {trial_args.lstm_layers}"
            )

            val_metrics = []
            best_epochs = []
            for val_year in val_years:
                set_seed(fixed_seed)
                # 滚动验证：用 val_year 之前的数据训练，用该年验证
                rs_train_mask = (years >= args.train_year_start) & (years <= val_year - 1)
                rs_val_mask = years == val_year
                if rs_train_mask.sum() == 0 or rs_val_mask.sum() == 0:
                    continue
                stats_rs = _compute_stats_for_mask(
                    x_all,
                    y_all,
                    static_all,
                    static_mask_all,
                    rs_train_mask,
                    args.use_dual_tower,
                )
                x_mean_rs, x_std_rs, y_mean_rs, y_std_rs, static_mean_rs, static_std_rs = stats_rs
                rs_train_loader = build_loader(
                    rs_train_mask,
                    trial_args.batch_size,
                    True,
                    stats_rs,
                )
                rs_val_loader = build_loader(
                    rs_val_mask,
                    trial_args.batch_size,
                    False,
                    stats_rs,
                )
                model, _, _, history, trial_best_epoch, _, _ = run_train_eval(
                    trial_args,
                    rs_train_loader,
                    rs_val_loader,
                    (x_mean_rs, x_std_rs, y_mean_rs, y_std_rs),
                    device,
                    save_prefix=None,
                    use_static=args.use_dual_tower,
                    static_stats=(static_mean_rs, static_std_rs) if args.use_dual_tower else None,
                    use_gnn=args.use_gnn,
                )
                if args.use_gnn:
                    val_rmse, val_mae, val_mape = eval_model_graph(
                        model,
                        rs_val_loader,
                        device,
                        y_mean_rs,
                        y_std_rs,
                        use_static=args.use_dual_tower,
                        compute_metrics=True,
                    )
                else:
                    val_rmse, val_mae, val_mape = eval_model(
                        model,
                        rs_val_loader,
                        device,
                        y_mean_rs,
                        y_std_rs,
                        use_static=args.use_dual_tower,
                        compute_metrics=True,
                    )
                val_metrics.append((val_rmse, val_mae, val_mape))
                best_epochs.append(trial_best_epoch)
                print(
                    f"验证年 {val_year} RMSE {val_rmse:.4f} MAE {val_mae:.4f} MAPE {val_mape:.2f}% "
                    f"(最佳轮次 {trial_best_epoch})"
                )
            if not val_metrics:
                continue
            val_rmse = sum(m[0] for m in val_metrics) / len(val_metrics)
            val_mae = sum(m[1] for m in val_metrics) / len(val_metrics)
            val_mape = sum(m[2] for m in val_metrics) / len(val_metrics)
            trial_best_epoch = int(round(sum(best_epochs) / max(1, len(best_epochs))))
            print(
                f"验证平均 RMSE {val_rmse:.4f} MAE {val_mae:.4f} MAPE {val_mape:.2f}% "
                f"(平均最佳轮次 {trial_best_epoch})"
            )

            if best_val_rmse is None or val_rmse < best_val_rmse:
                # 记录目前最好的试验结果
                best_val_rmse = val_rmse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_history = history
                best_val_metrics = (val_rmse, val_mae, val_mape, trial_best_epoch)
                best_params = {
                    "lr": trial_args.lr,
                    "batch_size": trial_args.batch_size,
                    "dropout": trial_args.dropout,
                    "transformer_heads": trial_args.transformer_heads,
                    "lstm_layers": trial_args.lstm_layers,
                }
                best_epoch = trial_best_epoch
                best_stats = stats_rs

        if best_state is None:
            raise RuntimeError("随机搜索没有产生有效试验")

        # 保存随机搜索得到的最佳模型
        save_prefix = os.path.join(args.out_dir, "random_search_best")
        best_model_args = get_model_args(
            args,
            args.use_dual_tower,
            overrides={
                "transformer_heads": best_params["transformer_heads"],
                "lstm_layers": best_params["lstm_layers"],
                "dropout": best_params["dropout"],
            },
        )
        save_checkpoint(
            f"{save_prefix}.pt",
            best_state,
            best_stats,
            args.use_dual_tower,
            best_model_args,
            extra={
                "params": best_params,
                "best_epoch": best_epoch,
                "val_rmse": best_val_metrics[0],
            },
        )
        print(
            f"最佳验证 RMSE {best_val_metrics[0]:.4f} MAE {best_val_metrics[1]:.4f} "
            f"MAPE {best_val_metrics[2]:.2f}% ，最佳轮次 {best_epoch}"
        )
        print(f"最佳模型已保存到 {os.path.abspath(f'{save_prefix}.pt')}")
        return
    else:
        # 不做随机搜索：仅进行滚动验证
        val_metrics = []
        best_epochs = []
        fold_histories = []
        year_points = {}
        results_path = os.path.join(args.out_dir, "results.csv")
        preds_path = os.path.join(args.out_dir, "rollval_predictions.csv")
        province_results_path = os.path.join(args.out_dir, "province_results.csv")
        province_preds_path = os.path.join(args.out_dir, "province_rollval_predictions.csv")
        province_result_fields = [
            "model_name",
            "val_year",
            "province_rmse",
            "province_mae",
            "province_mape",
            "n_val_samples",
        ]
        province_val_metrics = []
        if os.path.exists(preds_path):
            os.remove(preds_path)
        if args.enable_province_task and args.use_gnn:
            if os.path.exists(province_results_path):
                os.remove(province_results_path)
            if os.path.exists(province_preds_path):
                os.remove(province_preds_path)
        result_fields = [
            "model_name",
            "val_year",
            "best_val_rmse",
            "best_val_mae",
            "best_val_mape",
            "best_epoch",
            "seed",
            "n_train_samples",
            "n_val_samples",
        ]
        repr_year = val_years[len(val_years) // 2] if val_years else None
        repr_history = None
        repr_early_stop = None
        repr_best_epoch = None
        for val_year in val_years:
            set_seed(fixed_seed)
            roll_train_mask = (years >= args.train_year_start) & (years <= val_year - 1)
            roll_val_mask = years == val_year
            if roll_train_mask.sum() == 0 or roll_val_mask.sum() == 0:
                continue
            stats_roll = _compute_stats_for_mask(
                x_all,
                y_all,
                static_all,
                static_mask_all,
                roll_train_mask,
                args.use_dual_tower,
            )
            x_mean_roll, x_std_roll, y_mean_roll, y_std_roll, static_mean_roll, static_std_roll = stats_roll
            roll_train_loader = build_loader(
                roll_train_mask,
                args.batch_size,
                True,
                stats_roll,
            )
            roll_val_loader = build_loader(
                roll_val_mask,
                args.batch_size,
                False,
                stats_roll,
            )
            model, _, _, history, best_epoch, best_metrics, early_stop_epoch = run_train_eval(
                args,
                roll_train_loader,
                roll_val_loader,
                (x_mean_roll, x_std_roll, y_mean_roll, y_std_roll),
                device,
                save_prefix=None,
                use_static=args.use_dual_tower,
                static_stats=(static_mean_roll, static_std_roll) if args.use_dual_tower else None,
                use_gnn=args.use_gnn,
            )
            fold_histories.append(history)
            if best_metrics is not None:
                val_rmse, val_mae, val_mape = best_metrics
            elif args.use_gnn:
                val_rmse, val_mae, val_mape = eval_model_graph(
                    model,
                    roll_val_loader,
                    device,
                    y_mean_roll,
                    y_std_roll,
                    use_static=args.use_dual_tower,
                    compute_metrics=True,
                )
            else:
                val_rmse, val_mae, val_mape = eval_model(
                    model,
                    roll_val_loader,
                    device,
                    y_mean_roll,
                    y_std_roll,
                    use_static=args.use_dual_tower,
                    compute_metrics=True,
                )
            y_true, y_pred = collect_val_predictions(
                model,
                roll_val_loader,
                device,
                y_mean_roll,
                y_std_roll,
                use_static=args.use_dual_tower,
                use_gnn=args.use_gnn,
            )
            year_points[val_year] = (y_true, y_pred)
            val_metrics.append((val_rmse, val_mae, val_mape))
            best_epochs.append(best_epoch)
            print(f"验证年 {val_year} RMSE {val_rmse:.4f} MAE {val_mae:.4f} MAPE {val_mape:.2f}%")
            model_name = "graph" if args.use_gnn else ("dual" if args.use_dual_tower else "single")
            pred_rows = []
            roll_counties = counties[roll_val_mask]
            n_pred = min(len(roll_counties), len(y_true), len(y_pred))
            for county, y_true_i, y_pred_i in zip(
                roll_counties[:n_pred].tolist(),
                y_true[:n_pred].tolist(),
                y_pred[:n_pred].tolist(),
            ):
                pred_rows.append(
                    {
                        "model_name": model_name,
                        "val_year": int(val_year),
                        "county": str(county),
                        "y_true": float(y_true_i),
                        "y_pred": float(y_pred_i),
                        "error": float(y_pred_i - y_true_i),
                    }
                )
            append_county_prediction_rows(preds_path, pred_rows)
            append_results_row(
                results_path,
                {
                    "model_name": model_name,
                    "val_year": int(val_year),
                    "best_val_rmse": float(val_rmse),
                    "best_val_mae": float(val_mae),
                    "best_val_mape": float(val_mape),
                    "best_epoch": int(best_epoch) if best_epoch is not None else None,
                    "seed": int(fixed_seed),
                    "n_train_samples": int(roll_train_mask.sum()),
                    "n_val_samples": int(roll_val_mask.sum()),
                },
                result_fields,
            )
            if args.enable_province_task and args.use_gnn:
                p_true, p_pred = collect_graph_province_predictions(
                    model,
                    roll_val_loader,
                    device,
                    y_mean_roll,
                    y_std_roll,
                    use_static=args.use_dual_tower,
                )
                if p_true.size > 0 and p_pred.size > 0:
                    n_p = min(p_true.size, p_pred.size)
                    p_true = p_true[:n_p]
                    p_pred = p_pred[:n_p]
                    p_err = p_pred - p_true
                    p_rmse = float(np.sqrt(np.mean(p_err ** 2)))
                    p_mae = float(np.mean(np.abs(p_err)))
                    p_mape = float(np.mean(np.abs(p_err) / np.maximum(np.abs(p_true), 1e-6)) * 100.0)
                    province_val_metrics.append((p_rmse, p_mae, p_mape))
                    append_results_row(
                        province_results_path,
                        {
                            "model_name": model_name,
                            "val_year": int(val_year),
                            "province_rmse": p_rmse,
                            "province_mae": p_mae,
                            "province_mape": p_mape,
                            "n_val_samples": int(n_p),
                        },
                        province_result_fields,
                    )
                    pred_rows = []
                    for i in range(n_p):
                        pred_rows.append(
                            {
                                "model_name": model_name,
                                "val_year": int(val_year),
                                "y_true": float(p_true[i]),
                                "y_pred": float(p_pred[i]),
                                "error": float(p_err[i]),
                            }
                        )
                    append_province_prediction_rows(province_preds_path, pred_rows)
            if repr_year == val_year:
                repr_history = history
                repr_early_stop = early_stop_epoch
                repr_best_epoch = best_epoch
        if val_metrics:
            avg_rmse = sum(m[0] for m in val_metrics) / len(val_metrics)
            avg_mae = sum(m[1] for m in val_metrics) / len(val_metrics)
            avg_mape = sum(m[2] for m in val_metrics) / len(val_metrics)
            print(f"验证平均 RMSE {avg_rmse:.4f} MAE {avg_mae:.4f} MAPE {avg_mape:.2f}%")
            if province_val_metrics:
                p_avg_rmse = sum(m[0] for m in province_val_metrics) / len(province_val_metrics)
                p_avg_mae = sum(m[1] for m in province_val_metrics) / len(province_val_metrics)
                p_avg_mape = sum(m[2] for m in province_val_metrics) / len(province_val_metrics)
                print(
                    f"省级验证平均 RMSE {p_avg_rmse:.4f} MAE {p_avg_mae:.4f} MAPE {p_avg_mape:.2f}% | "
                    f"results={province_results_path}"
                )
            rmse_histories = [h["val_rmse"] for h in fold_histories if h.get("val_rmse")]
            rmse_path = os.path.join(args.out_dir, "rollval_rmse_convergence.png")
            save_rollval_rmse_plot(rmse_histories, rmse_path)
            if repr_history is not None:
                repr_path = os.path.join(args.out_dir, f"rollval_{repr_year}_rmse.png")
                save_repr_rmse_plot(
                    repr_history.get("val_rmse", []),
                    repr_path,
                    repr_best_epoch,
                    repr_early_stop,
                )
            if year_points:
                scatter_path = os.path.join(args.out_dir, "rollval_scatter.png")
                save_rollval_scatter_plot(year_points, scatter_path)
        else:
            raise RuntimeError("滚动验证没有产生有效的验证结果")

if __name__ == "__main__":
    main()
