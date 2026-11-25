# -*- coding: utf-8 -*-
"""
全国SIF(YYYYDDD.tif) -> 裁剪到吉林bbox -> (县域 ∩ 玉米区) 掩膜 -> 按县统计 SIF
输出：Jilin_{year}_SIF_Stats_Full.csv

依赖：
pip install xarray rioxarray geopandas rasterio tqdm pandas numpy
"""

import os
import glob
import re
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
from tqdm import tqdm
from rasterio import features
from rasterio.enums import Resampling

warnings.filterwarnings("ignore")

# ================= 1. 路径与参数配置 =================

# (1) 全国 SIF tif 根目录：里面按年份分文件夹，例如 ...\geotiff\2001\2001060.tif
SIF_ROOT = r"E:\data\sif\geotiff"

# (2) 玉米分布图总根目录（每年一个文件夹）
CCD_MAIZE_ROOT = r"E:\data\CCD-Maize"

# (3) 吉林省县级边界（GeoJSON / Shapefile）
COUNTY_SHP_PATH = r"E:\data\吉林省.json"

# (4) 结果保存路径
OUTPUT_DIR = r"E:\data\sif\Final_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (5) 年份范围
START_YEAR = 2002
END_YEAR = 2019

# (6) 并行核数
MAX_WORKERS = 6

# (7) SIF 缩放因子
SIF_SCALE = 0.0001

# (8) 玉米图阈值策略：取 >0 的最大值作为“玉米类别”
#     如果你的玉米图类别不是这样，后续可以改这里
# ================= 2. 工具函数 =================

def find_corn_file(year: int):
    """
    寻找玉米图
    兼容类似：*Jilin-maize-{year}*.tif
    """
    year_folder = os.path.join(CCD_MAIZE_ROOT, str(year))
    if not os.path.exists(year_folder):
        print(f"⚠️ 警告：找不到年份文件夹: {year_folder}")
        return None

    pattern = os.path.join(year_folder, f"*Jilin-maize-{year}*.tif")
    found_files = glob.glob(pattern)

    if not found_files:
        print(f"⚠️ 警告：在 {year} 年里没找到吉林玉米图！(pattern={pattern})")
        return None

    return found_files[0]


def parse_yyyydoy_key_from_filename(path: str) -> str:
    """
    文件名：YYYYDDD.tif 例如 2001060.tif -> '2001060'
    """
    base = os.path.splitext(os.path.basename(path))[0]
    if re.fullmatch(r"\d{7}", base):
        return base
    raise ValueError(f"文件名不是YYYYDDD格式: {os.path.basename(path)}")


def yyyydoy_to_datestr(yyyydoy: str) -> str:
    """
    '2001060' -> '2001-03-01'
    """
    year = int(yyyydoy[:4])
    doy = int(yyyydoy[4:])
    d = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return d.strftime("%Y-%m-%d")


def build_date_to_path_for_year(year: int):
    """
    扫描某年的 SIF tif，生成 date_key(YYYYDDD) -> filepath 映射
    """
    year_folder = os.path.join(SIF_ROOT, str(year))
    sif_files = glob.glob(os.path.join(year_folder, "*.tif"))
    if not sif_files:
        return {}, year_folder

    date_to_path = {}
    for f in sif_files:
        try:
            k = parse_yyyydoy_key_from_filename(f)
            date_to_path[k] = f
        except Exception as e:
            print(f"跳过无法解析的文件: {os.path.basename(f)} -> {e}")

    return date_to_path, year_folder


def get_valid_indices_and_ids(template_sif_path: str, county_shp_path: str, corn_map_path: str):
    """
    关键：全国模板SIF -> 先裁到吉林bbox -> 在吉林窗口上：
    1) rasterize 县ID栅格
    2) 玉米图 reproject_match 到该窗口
    3) 联合掩膜 (县内 & 玉米)
    返回：
      valid_indices:  吉林窗口flatten后的有效像元索引
      valid_county_ids: 对应的县id
      id_to_name: 县id->县名
      bbox: 吉林边界在SIF CRS下的 (minx, miny, maxx, maxy)
    """
    print(f"   -> 读取模板(全国SIF): {os.path.basename(template_sif_path)}")

    # 1) 读全国模板
    sif_full = rioxarray.open_rasterio(template_sif_path, masked=True).squeeze()
    target_crs = sif_full.rio.crs
    if target_crs is None:
        raise ValueError("模板SIF没有CRS信息，请先确保tif带投影。")

    # 2) 读吉林县界并投影到 SIF CRS
    gdf = gpd.read_file(county_shp_path)
    if gdf.crs is None:
        raise ValueError("县界文件没有CRS信息，请先确保边界文件带投影。")

    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)

    # 3) 用吉林外包框裁剪模板到吉林窗口（巨幅提速/省内存）
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = (minx, miny, maxx, maxy)

    sif_win = sif_full.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    sif_full.close()

    template_shape = sif_win.shape
    template_transform = sif_win.rio.transform()
    real_x = sif_win.x
    real_y = sif_win.y

    # 4) 县id映射（县名字段按你的json里是 'name'；如不是请改字段名）
    gdf = gdf.copy()
    gdf["temp_id"] = range(1, len(gdf) + 1)
    if "name" not in gdf.columns:
        raise ValueError("县界属性表里找不到 'name' 字段，请把字段名改成你的县名字段。")
    id_to_name = gdf.set_index("temp_id")["name"].to_dict()

    # 5) rasterize 县网格（只在吉林窗口）
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["temp_id"]))
    county_grid = features.rasterize(
        shapes=shapes,
        out_shape=template_shape,
        transform=template_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint16
    )

    # 6) 玉米图对齐到吉林窗口
    with rioxarray.open_rasterio(corn_map_path, masked=True) as corn_da:
        # 用吉林窗口做 match 模板
        template_ds = xr.DataArray(
            data=np.zeros(template_shape, dtype=np.uint8),
            dims=("y", "x"),
            coords={"y": real_y, "x": real_x}
        )
        template_ds.rio.write_crs(target_crs, inplace=True)
        template_ds.rio.write_transform(template_transform, inplace=True)

        corn_aligned = corn_da.squeeze().rio.reproject_match(template_ds, resampling=Resampling.nearest)

        # 取 >0 的最大值作为玉米类别
        unique_vals = np.unique(corn_aligned.values)
        valid_vals = unique_vals[unique_vals > 0]
        if len(valid_vals) == 0:
            raise ValueError("玉米图在吉林窗口内全是0/无效！")
        target_val = valid_vals.max()
        corn_mask = (corn_aligned.values == target_val)

    # 7) 联合掩膜：县内 & 玉米
    final_mask = (county_grid > 0) & corn_mask
    valid_indices = np.where(final_mask.flatten())[0]
    valid_county_ids = county_grid.flatten()[valid_indices]

    if len(valid_indices) == 0:
        raise ValueError("❌ 有效像素为 0：请检查玉米图是否正确、以及SIF/边界投影是否匹配。")

    return valid_indices, valid_county_ids, id_to_name, bbox


def process_one_date(date_key: str,
                     date_to_path: dict,
                     valid_indices: np.ndarray,
                     valid_county_ids: np.ndarray,
                     id_to_name: dict,
                     bbox: tuple):
    """
    单期(8天) SIF 数据统计
    date_key: '2001060'
    """
    daily_df = pd.DataFrame({"temp_id": list(id_to_name.keys())})
    daily_df["日期"] = date_key
    daily_df["日期_可读"] = yyyydoy_to_datestr(date_key)
    daily_df["县名"] = daily_df["temp_id"].map(id_to_name)
    daily_df.set_index("temp_id", inplace=True)
    daily_df.index = daily_df.index.astype(int)

    def p10(x): return x.quantile(0.1)
    def p90(x): return x.quantile(0.9)

    out_cols = ["SIF_median", "SIF_std", "SIF_p10", "SIF_p90"]
    file_path = date_to_path[date_key]

    try:
        with rioxarray.open_rasterio(file_path, masked=True) as src:
            da = src.squeeze()

            # 裁剪到吉林窗口（必须与 valid_indices 的窗口一致）
            da = da.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

            arr = da.values.astype(float)

            # nodata兜底（masked=True 一般已经处理，但这里更稳）
            nodata = da.rio.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan

            # 缩放因子
            arr *= SIF_SCALE

            data_flat = arr.flatten()

        valid_data = data_flat[valid_indices]
        mask_valid = ~np.isnan(valid_data)

        if np.any(mask_valid):
            current_ids = valid_county_ids[mask_valid]
            current_vals = valid_data[mask_valid]

            df_temp = pd.DataFrame({"id": current_ids, "val": current_vals})
            stats = df_temp.groupby("id")["val"].agg(["median", "std", p10, p90])
            stats.columns = out_cols
            stats.index = stats.index.astype(int)

            daily_df = daily_df.join(stats)
        else:
            for c in out_cols:
                daily_df[c] = np.nan

    except Exception as e:
        print(f"❌ {date_key} 处理失败: {os.path.basename(file_path)} -> {e}")
        for c in out_cols:
            daily_df[c] = np.nan

    return daily_df.reset_index(drop=True)


# ================= 3. 主程序 =================

def main():
    print("🚀 启动 SIF(全国) -> 吉林(县域∩玉米区) 统计程序")
    print(f"📂 SIF_ROOT: {SIF_ROOT}")
    print(f"📂 CCD_MAIZE_ROOT: {CCD_MAIZE_ROOT}")
    print(f"🗺️ COUNTY_SHP_PATH: {COUNTY_SHP_PATH}")
    print(f"📅 年份范围: {START_YEAR} - {END_YEAR}")
    print(f"⚙️ 并行核数: {MAX_WORKERS}")
    print(f"🔧 SIF 缩放因子: {SIF_SCALE}")

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n{'='*60}")
        print(f"🌽 处理年份: {year}")
        print(f"{'='*60}")

        # A) 读取该年SIF文件
        date_to_path, year_folder = build_date_to_path_for_year(year)
        if not date_to_path:
            print(f"⚠️ 跳过：{year_folder} 没有可用的SIF tif")
            continue

        dates = sorted(date_to_path.keys())
        print(f"   -> 找到 {len(dates)} 个8天SIF文件 (例如: {dates[0]} ...)")

        # B) 找玉米图
        corn_file = find_corn_file(year)
        if not corn_file:
            print(f"⚠️ 跳过：没找到 {year} 年的玉米图。")
            continue
        print(f"   -> 玉米图: {os.path.basename(corn_file)}")

        # C) 构建掩膜（用该年第一张SIF当模板）
        template_tif = date_to_path[dates[0]]
        try:
            valid_indices, valid_county_ids, id_to_name, bbox = get_valid_indices_and_ids(
                template_tif, COUNTY_SHP_PATH, corn_file
            )
            print(f"   -> 有效像元数(吉林玉米区): {len(valid_indices)}")
        except Exception as e:
            print(f"❌ 掩膜构建失败: {e}")
            continue

        # D) 并行统计
        print(f"🔥 开始并行统计 {year} 年...")
        results = []

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_one_date,
                    date_key,
                    date_to_path,
                    valid_indices,
                    valid_county_ids,
                    id_to_name,
                    bbox
                ): date_key
                for date_key in dates
            }

            for future in tqdm(as_completed(futures), total=len(dates), desc=f"{year}进度"):
                date_key = futures[future]
                try:
                    df = future.result()
                    results.append(df)
                except Exception as e:
                    print(f"❌ {date_key} 失败: {e}")

        # E) 保存
        if results:
            full_df = pd.concat(results, ignore_index=True)
            full_df = full_df.sort_values(by=["日期", "县名"])

            # 列顺序
            base_cols = ["日期", "日期_可读", "县名"]
            other_cols = [c for c in full_df.columns if c not in base_cols and c != "temp_id"]
            full_df = full_df[base_cols + other_cols]

            save_name = f"Jilin_{year}_SIF_Stats_Full.csv"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            full_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"🎉 {year} 年完成！已保存: {save_path}")
        else:
            print(f"⚠️ {year} 年无结果数据。")

    print("\n🏁 全部任务结束！")


if __name__ == "__main__":
    main()
