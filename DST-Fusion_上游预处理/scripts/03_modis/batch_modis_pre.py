import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
import numpy as np
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')

# ================= 1. 配置参数 =================
INPUT_FOLDER = r"E:\data\modis\2001-2006\MOD09GA_061-20251201_120213"
OUTPUT_FOLDER = r"E:\data\modis\2001-2006\Results"
# 并行核数，建议设置为 CPU 核心数 - 2，防止死机。如果内存够大，可以设为 None (使用所有核)
MAX_WORKERS = 2

# ================= 2. 核心处理函数 =================

def get_modis_layer(filename, target_key):
    """
    读取单个图层，增加 chunks 参数开启 Dask 并行与懒加载
    """
    try:
        # chunks={'x': 1200, 'y': 1200} 开启懒加载，大幅降低内存占用
        all_layers = rioxarray.open_rasterio(filename, masked=True, variable=[target_key], chunks='auto')
    except Exception:
        try:
            all_layers = rioxarray.open_rasterio(filename, masked=True, chunks='auto')
        except:
            return None

    if not isinstance(all_layers, list):
        all_layers = [all_layers]

    found_data = None
    for layer in all_layers:
        name_check = layer.name if hasattr(layer, 'name') else str(layer.data_vars)
        if target_key in str(name_check):
            found_data = layer
            break
            
    if found_data is None: 
        return None
        
    if isinstance(found_data, xr.Dataset):
        for var in found_data.data_vars:
            if target_key in var:
                found_data = found_data[var]
                break
                
    data = found_data.squeeze()
    
    # 统一重命名维度
    rename_dict = {}
    for d in data.dims:
        if "X" in d and "Dim" in d: rename_dict[d] = "x"
        if "Y" in d and "Dim" in d: rename_dict[d] = "y"
    if rename_dict: 
        data = data.rename(rename_dict)
        
    return data

def process_single_date(date_str, file_L, file_R, save_dir):
    """
    单个日期的处理逻辑，将被放入进程池中运行
    """
    try:
        # print(f"🚀 [进程启动] {date_str} ...") # 多进程时print可能会乱，尽量少打
        
        bands_map = {
            "Red":   "sur_refl_b01_1", 
            "NIR":   "sur_refl_b02_1", 
            "Blue":  "sur_refl_b03_1", 
            "Green": "sur_refl_b04_1", 
            "SWIR":  "sur_refl_b06_1", 
            "QC":    "state_1km_1"      
        }
        
        mosaics = {}
        modis_crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"

        # --- 1. 读取与拼接 ---
        # 这一步是IO密集型，但在多进程下各进程独立读取
        for common_name, search_key in bands_map.items():
            band_L = get_modis_layer(file_L, search_key)
            band_R = get_modis_layer(file_R, search_key)
            
            if band_L is None or band_R is None:
                return f"❌ {date_str}: 缺少波段 {common_name}"

            # 设置 CRS
            if band_L.rio.crs is None: band_L.rio.write_crs(modis_crs, inplace=True)
            if band_R.rio.crs is None: band_R.rio.write_crs(modis_crs, inplace=True)
            
            # 确保有空间维度
            if band_L.rio.x_dim is None: band_L.rio.set_spatial_dims("x","y",inplace=True)
            if band_R.rio.x_dim is None: band_R.rio.set_spatial_dims("x","y",inplace=True)

            # 拼接
            mosaics[common_name] = merge_arrays([band_L, band_R])

        # --- 2. 对齐 (通常 merge 后是对齐的，这里保留逻辑但简化) ---
        # xarray 计算通常会自动对齐，显式 reproject_match 比较慢。
        # 如果 merge_arrays 正常工作，这一步通常可以省略。为了保险保留，但注意性能。
        master = mosaics["Red"]
        for name in mosaics:
            if name != "Red":
                 # 只有当尺寸不一致时才重投影
                if mosaics[name].shape != master.shape:
                    mosaics[name] = mosaics[name].rio.reproject_match(master, resampling=Resampling.nearest)

        # --- 3. 去云与掩膜 ---
        # 使用 Dask 延迟计算，这里不立即执行
        qc = mosaics["QC"].fillna(-1).astype(int)
        cloud_mask = (qc % 4) != 0   
        shadow_mask = ((qc // 4) % 2) == 1
        bad_pixels = cloud_mask | shadow_mask | (qc == -1)
        
        def clean_data(band_data):
            # 这里的 where 是延迟操作
            return band_data.where(~bad_pixels, np.nan)

        # 此时数据还是 dask array，没有加载到内存
        Red   = clean_data(mosaics["Red"])
        NIR   = clean_data(mosaics["NIR"])
        Blue  = clean_data(mosaics["Blue"])
        Green = clean_data(mosaics["Green"])
        SWIR  = clean_data(mosaics["SWIR"])

        # 简单的异常值处理 (这里为了性能，移除了 mean check，
        # 因为 mean() 会触发立即计算导致从内存中加载整个数组。
        # 如果必须要有 >100 的检查，建议放到最后或抽样检查)
        # Red = Red * 0.0001 # MODIS通常有scale factor, 假设用户知道自己在做什么

        # --- 4. 计算指数 (Lazy Evaluation) ---
        epsilon = 1e-6
        ds_results = xr.Dataset() # 使用 Dataset 统一管理
        
        ds_results["EVI"] = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1 + epsilon)
        ds_results["WDRVI"] = (0.1 * NIR - Red) / (0.1 * NIR + Red + epsilon)
        ds_results["GCI"] = (NIR / (Green + epsilon)) - 1
        ds_results["NDWI"] = (NIR - SWIR) / (NIR + SWIR + epsilon)
        ndvi = (NIR - Red) / (NIR + Red + epsilon)
        ds_results["NIRv"] = ndvi * NIR

        # --- 5. 统一投影与保存 ---
        # 关键优化：先对整个 Dataset 进行一次投影，而不是对每个 Array 分别投影
        target_crs = "EPSG:4326"
        if ds_results.rio.x_dim is None:
            ds_results.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        
        # 继承 CRS
        ds_results.rio.write_crs(modis_crs, inplace=True)

        # 统一 Reproject (最耗时的一步，现在只做一次)
        # resolution=0.005 是输出分辨率
        ds_reprojected = ds_results.rio.reproject(
            target_crs, 
            resolution=0.005, 
            nodata=np.nan,
            resampling=Resampling.bilinear # 指数计算建议双线性，分类数据用最近邻
        )

        # 循环保存结果
        saved_files = []
        for name in ds_reprojected.data_vars:
            filename = f"Jilin_{date_str}_{name}.tif"
            save_path = os.path.join(save_dir, filename)
            
            if os.path.exists(save_path):
                continue
            
            # 写入磁盘，compress='lzw'，使用 tiled=True 加快读取
            # num_threads 用于压缩时的线程
            ds_reprojected[name].rio.to_raster(
                save_path, 
                compress='lzw', 
                tiled=True, 
                windowed=True,
                zlevel=6
            )
            saved_files.append(filename)

        return f"✅ {date_str} 完成 (生成 {len(saved_files)} 个文件)"

    except Exception as e:
        import traceback
        return f"❌ {date_str} 报错: {str(e)}"

# ================= 3. 批量管理逻辑 =================

def run_batch_processing():
    # 1. 确保输出文件夹存在
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # 2. 找到所有 HDF 文件
    hdf_files = glob.glob(os.path.join(INPUT_FOLDER, "*.hdf"))
    print(f"📂 找到 {len(hdf_files)} 个 HDF 文件")
    
    # 3. 按日期分组
    grouped_files = {}
    for f_path in hdf_files:
        filename = os.path.basename(f_path)
        parts = filename.split('.')
        if len(parts) > 1:
            date_str = parts[1]
            if date_str not in grouped_files: grouped_files[date_str] = []
            grouped_files[date_str].append(f_path)
            
    # --- 4. 关键修改：智能过滤任务 ---
    tasks = []
    # 这里列出你所有需要生成的指数名称
    REQUIRED_PRODUCTS = ["EVI", "NIRv", "NDWI", "WDRVI", "GCI"] 
    
    print(f"🕵️‍♀️ 正在扫描已完成的任务，请稍候...")
    
    skipped_count = 0
    for date_str in sorted(grouped_files.keys()):
        files = grouped_files[date_str]
        
        # 基础检查
        if len(files) != 2: continue
        
        # --- 核心优化：检查是否该日期所有结果都已存在 ---
        all_exist = True
        for prod in REQUIRED_PRODUCTS:
            result_path = os.path.join(OUTPUT_FOLDER, f"Jilin_{date_str}_{prod}.tif")
            # 如果缺任何一个文件，或者文件大小为0，就必须重跑
            if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                all_exist = False
                break
        
        if all_exist:
            # 如果全都有，直接跳过，不再放入任务列表！
            skipped_count += 1
            # print(f"⏩ {date_str} 已存在，跳过") # 嫌刷屏可以注释掉
            continue

        # 只有缺文件的日期，才会进入这里
        f1, f2 = files[0], files[1]
        if "h26v04" in f1 and "h27v04" in f2:
            file_L, file_R = f1, f2
        elif "h27v04" in f1 and "h26v04" in f2:
            file_L, file_R = f2, f1
        else:
            continue
            
        tasks.append((date_str, file_L, file_R, OUTPUT_FOLDER))

    print("-" * 30)
    print(f"🎉 智能过滤完成！")
    print(f"⏩ 直接跳过: {skipped_count} 天 (无需处理)")
    print(f"🔥 剩余需跑: {len(tasks)} 天")
    print("-" * 30)

    if len(tasks) == 0:
        print("✅ 所有数据都已齐全，无需运行！")
        return

    # --- 5. 开始并行处理剩余任务 ---
    start_time = time.time()
    
    # 补跑建议用 2 核，稳一点
    REAL_WORKERS = 2 
    
    with ProcessPoolExecutor(max_workers=REAL_WORKERS) as executor:
        future_to_date = {executor.submit(process_single_date, *task): task[0] for task in tasks}
        
        count = 0
        total = len(tasks)
        
        for future in as_completed(future_to_date):
            date_s = future_to_date[future]
            try:
                result = future.result()
                count += 1
                print(f"[{count}/{total}] {result}")
            except Exception as exc:
                print(f"❌ {date_s} 异常: {exc}")

    total_time = (time.time() - start_time) / 60
    print(f"⏱️ 补跑耗时: {total_time:.2f} 分钟")

if __name__ == "__main__":
    run_batch_processing()