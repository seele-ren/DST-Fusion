import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import regionmask
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ================= 1. 配置区 =================
cmfd_dir = r"E:\data\cmfd"
tmax_dir = r"E:\data\cmfd\tmax"
json_path = r"E:\data\吉林省.json"
output_dir = r"E:\data\result_csv_final"
os.makedirs(output_dir, exist_ok=True)

years_to_process = range(2001, 2020)
county_col_name = 'name'

# ================= 2. 核心计算函数 =================
def process_group(gdf, ds_merged, group_name):
    # 双重检查：如果数据切出来是空的，直接返回空表，防止 regionmask 报错
    if ds_merged.lat.size == 0 or ds_merged.lon.size == 0:
        # tqdm.write(f"⚠️ {group_name} 切片为空，可能是坐标范围不匹配")
        return pd.DataFrame()

    try:
        # 1. 制作模具
        mask = regionmask.Regions(gdf.geometry, names=gdf.county_id).mask(
            ds_merged.lon, ds_merged.lat
        )
        mask.name = 'region'

        # 2. 计算平均值
        county_means = ds_merged.groupby(mask).mean()
        
        # 3. 转 DataFrame
        id_to_name = gdf.set_index('county_id')[county_col_name].to_dict()
        df_list = []
        
        for region_id in county_means.region.values:
            ds_county = county_means.sel(region=region_id)
            df = ds_county.to_dataframe().reset_index()
            real_name = id_to_name.get(int(region_id), f"Unknown_{int(region_id)}")
            df['County'] = real_name
            df_list.append(df)
        
        if not df_list:
            return pd.DataFrame()

        full_df = pd.concat(df_list)
        return full_df
        
    except ValueError as e:
        # 捕捉 "region must not be empty" 这种特定错误
        # tqdm.write(f"⚠️ {group_name} 掩膜生成失败 (可能该年数据未覆盖吉林省): {e}")
        return pd.DataFrame()

def calculate_vpd(temp_c, pres_pa, shum_kgkg):
    es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3)) 
    ea = (shum_kgkg * pres_pa) / (0.622 + 0.378 * shum_kgkg) / 1000.0 
    return np.maximum(es - ea, 0)

# ================= 3. 主程序 =================
def main():
    print("🚀 程序启动，正在处理...")
    
    # 1. 准备地图
    gdf = gpd.read_file(json_path)
    if gdf.crs is None: gdf.set_crs("epsg:4326", inplace=True)
    gdf['county_id'] = range(len(gdf))
    minx, miny, maxx, maxy = gdf.total_bounds

    for year in tqdm(years_to_process, desc="进度", unit="年"):
        
        # ================= 第一步：处理 CMFD =================
        cmfd_files = {
            "temp": os.path.join(cmfd_dir, f"temp_CMFD_V0200_B-01_01dy_010deg_{year}01-{year}12.nc"),
            "prec": os.path.join(cmfd_dir, f"prec_CMFD_V0200_B-01_01dy_010deg_{year}01-{year}12.nc"),
            "pres": os.path.join(cmfd_dir, f"pres_CMFD_V0200_B-01_01dy_010deg_{year}01-{year}12.nc"),
            "shum": os.path.join(cmfd_dir, f"shum_CMFD_V0200_B-01_01dy_010deg_{year}01-{year}12.nc"),
        }
        
        if any([not os.path.exists(p) for p in cmfd_files.values()]):
            continue

        try:
            ds_list = []
            for var_name, path in cmfd_files.items():
                ds = xr.open_dataset(path)
                # 改名
                if 'longitude' in ds.dims: ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
                
                # ⭐️【关键修复 1】：不管原来是不是乱序，强制从小到大排序
                ds = ds.sortby(['lat', 'lon'])
                
                # 抹平时间
                if 'time' in ds.coords: ds['time'] = ds.time.dt.floor('D')
                
                # 切片 (现在排好序了，切片绝对安全)
                ds = ds.sel(lon=slice(minx-0.1, maxx+0.1), lat=slice(miny-0.1, maxy+0.1))
                ds_list.append(ds)
            
            ds_cmfd = xr.merge(ds_list)
            df_cmfd = process_group(gdf, ds_cmfd, "CMFD组")
            ds_cmfd.close()

            # ================= 第二步：处理 Tmax =================
            tmax_path = os.path.join(tmax_dir, f"{year}.nc")
            df_tmax = pd.DataFrame() # 默认为空

            if os.path.exists(tmax_path):
                ds_tmax = xr.open_dataset(tmax_path)
                if 'longitude' in ds_tmax.dims: ds_tmax = ds_tmax.rename({'longitude': 'lon', 'latitude': 'lat'})
                
                # ⭐️【关键修复 2】：这里最重要！你的 Tmax 纬度是反的，这句能救命
                ds_tmax = ds_tmax.sortby(['lat', 'lon'])
                
                if 'time' in ds_tmax.coords: ds_tmax['time'] = ds_tmax.time.dt.floor('D')
                
                # 切片
                ds_tmax = ds_tmax.sel(lon=slice(minx-0.1, maxx+0.1), lat=slice(miny-0.1, maxy+0.1))
                
                # 检查切片后是否还有数据
                if ds_tmax.lat.size > 0 and ds_tmax.lon.size > 0:
                    df_tmax = process_group(gdf, ds_tmax, "Tmax组")
                
                ds_tmax.close()

            # ================= 第三步：合并与保存 =================
            if df_cmfd.empty:
                tqdm.write(f"⚠️ {year} 年 CMFD 数据处理结果为空，跳过")
                continue

            if not df_tmax.empty:
                # 合并
                final_df = pd.merge(df_cmfd, df_tmax[['County', 'time', 'tmax']], on=['County', 'time'], how='left')
            else:
                final_df = df_cmfd
                final_df['tmax'] = np.nan

            # 变量名检查
            if 'temp' not in final_df.columns:
                for n in ['tmp', 'temperature', 't2m']:
                    if n in final_df.columns: final_df.rename(columns={n: 'temp'}, inplace=True); break
            
            final_df = final_df.dropna(subset=['temp'])
            
            final_df['temp_avg_c'] = final_df['temp'] - 273.15
            
            if 'tmax' in final_df.columns:
                final_df['tmax_c'] = final_df['tmax'] 
            else:
                final_df['tmax_c'] = np.nan

            final_df['GDD'] = np.maximum(final_df['temp_avg_c'] - 10, 0)
            final_df['KDD'] = np.maximum(final_df['tmax_c'] - 29, 0)
            final_df['VPD'] = calculate_vpd(final_df['temp_avg_c'], final_df['pres'], final_df['shum'])
            final_df['PRCP'] = final_df['prec'] * 86400 

            save_name = f"吉林_气候数据_{year}.csv"
            save_path = os.path.join(output_dir, save_name)
            
            cols = ['County', 'time', 'temp_avg_c', 'tmax_c', 'GDD', 'KDD', 'PRCP', 'VPD']
            valid_cols = [c for c in cols if c in final_df.columns]
            
            final_df[valid_cols].to_csv(save_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            tqdm.write(f"❌ {year} 年出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 全部结束！")

if __name__ == "__main__":
    main()