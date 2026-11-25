import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from rasterio import features
from rasterio.enums import Resampling
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")

INDICATORS = ["EVI", "WDRVI", "GCI", "NDWI", "NIRv"]
STAT_SUFFIXES = ["median", "std", "p10", "p90"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract county-level MODIS index statistics with a year-matched maize mask."
    )
    parser.add_argument("--province", default="Jilin", help="Output/file prefix, e.g. Jilin or Liaoning.")
    parser.add_argument("--modis-folder", default=r"E:\data\modis\2001-2006\Results")
    parser.add_argument("--ccd-maize-root", default=r"E:\data\CCD-Maize")
    parser.add_argument("--county", default=r"E:\data\吉林省.json", help="County boundary GeoJSON/SHP.")
    parser.add_argument("--output-dir", default=r"E:\data\modis\Final_Results")
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2001)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--name-column", default="name", help="County-name column in boundary file.")
    parser.add_argument("--code-column", default="adcode", help="Optional county-code column in boundary file.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_corn_file(year, ccd_maize_root, province):
    year_folder = os.path.join(ccd_maize_root, str(year))
    if not os.path.exists(year_folder):
        print(f"Warning: missing maize-mask year folder: {year_folder}")
        return None

    patterns = [
        os.path.join(year_folder, f"*{province}-maize-{year}*.tif"),
        os.path.join(year_folder, f"*{province}*maize*{year}*.tif"),
    ]
    for pattern in patterns:
        found_files = glob.glob(pattern)
        if found_files:
            return found_files[0]

    print(f"Warning: no {province} maize mask found for {year} in {year_folder}")
    return None


def get_county_names(gdf, name_column):
    if name_column in gdf.columns:
        return gdf[name_column]

    fallback_columns = ["NAME", "Name", "县名", "县", "区县", "市县"]
    for column in fallback_columns:
        if column in gdf.columns:
            print(f"Using boundary name column: {column}")
            return gdf[column]

    raise ValueError(
        f"Boundary file has no county-name column '{name_column}'. "
        f"Available columns: {list(gdf.columns)}"
    )


def get_county_codes(gdf, code_column):
    if code_column and code_column in gdf.columns:
        return gdf[code_column].astype(str)
    return None


def get_valid_indices_and_ids(tif_template_path, county_path, corn_map_path, name_column, code_column):
    print(f"   -> Reading maize mask: {os.path.basename(corn_map_path)}")

    try:
        modis_ds = rioxarray.open_rasterio(tif_template_path, masked=True).squeeze()
    except Exception as exc:
        raise ValueError(f"Failed to read MODIS template: {exc}")

    target_crs = modis_ds.rio.crs
    template_shape = modis_ds.shape
    template_transform = modis_ds.rio.transform()
    real_x = modis_ds.x
    real_y = modis_ds.y
    modis_ds.close()

    gdf = gpd.read_file(county_path)
    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)

    gdf = gdf.copy()
    gdf["temp_id"] = range(1, len(gdf) + 1)
    gdf["_county_name"] = get_county_names(gdf, name_column).astype(str)
    county_codes = get_county_codes(gdf, code_column)
    if county_codes is not None:
        gdf["_county_code"] = county_codes
    id_to_name = gdf.set_index("temp_id")["_county_name"].to_dict()
    id_to_code = (
        gdf.set_index("temp_id")["_county_code"].to_dict()
        if "_county_code" in gdf.columns
        else {}
    )

    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["temp_id"]))
    county_grid = features.rasterize(
        shapes=shapes,
        out_shape=template_shape,
        transform=template_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint16,
    )

    with rioxarray.open_rasterio(corn_map_path, masked=True) as corn_da:
        template_ds = xr.DataArray(
            data=np.zeros(template_shape),
            dims=("y", "x"),
            coords={"y": real_y, "x": real_x},
        )
        template_ds.rio.write_crs(target_crs, inplace=True)
        template_ds.rio.write_transform(template_transform, inplace=True)

        corn_aligned = corn_da.squeeze().rio.reproject_match(
            template_ds, resampling=Resampling.nearest
        )

        unique_vals = np.unique(corn_aligned.values)
        valid_vals = unique_vals[unique_vals > 0]
        if len(valid_vals) == 0:
            raise ValueError("Maize mask contains no positive pixels.")

        target_val = valid_vals.max()
        corn_mask = corn_aligned.values == target_val

    final_mask = (county_grid > 0) & corn_mask
    valid_indices = np.where(final_mask.flatten())[0]
    valid_county_ids = county_grid.flatten()[valid_indices]

    if len(valid_indices) == 0:
        raise ValueError("No valid pixels after county boundary and maize mask intersection.")

    print(f"   -> Valid maize pixels: {len(valid_indices)}")
    return valid_indices, valid_county_ids, id_to_name, id_to_code


def p10(x):
    return x.quantile(0.1)


def p90(x):
    return x.quantile(0.9)


def process_one_date(date_str, folder, province, valid_indices, valid_county_ids, id_to_name, id_to_code):
    daily_df = pd.DataFrame({"temp_id": list(id_to_name.keys())})
    daily_df["日期"] = date_str
    if id_to_code:
        daily_df["区县代码"] = daily_df["temp_id"].map(id_to_code)
    daily_df["县名"] = daily_df["temp_id"].map(id_to_name)
    daily_df.set_index("temp_id", inplace=True)
    daily_df.index = daily_df.index.astype(int)

    for indicator in INDICATORS:
        target_cols = [f"{indicator}_{suffix}" for suffix in STAT_SUFFIXES]
        filename = f"{province}_{date_str}_{indicator}.tif"
        file_path = os.path.join(folder, filename)

        calc_success = False
        if os.path.exists(file_path):
            try:
                with rioxarray.open_rasterio(file_path, masked=True) as src:
                    data_flat = src.squeeze().values.flatten()

                valid_data = data_flat[valid_indices]
                mask_valid = ~np.isnan(valid_data)

                if np.any(mask_valid):
                    current_ids = valid_county_ids[mask_valid]
                    current_vals = valid_data[mask_valid]

                    df_temp = pd.DataFrame({"id": current_ids, "val": current_vals})
                    df_temp["val"] = df_temp["val"].astype(float)

                    stats = df_temp.groupby("id")["val"].agg(["median", "std", p10, p90])
                    stats.columns = target_cols
                    stats.index = stats.index.astype(int)

                    daily_df = daily_df.join(stats)
                    calc_success = True
            except Exception:
                pass

        if not calc_success:
            for col in target_cols:
                daily_df[col] = np.nan

    return daily_df.reset_index(drop=True)


def list_year_dates(all_tif_files, year, province):
    target_prefix = f"A{year}"
    dates = []
    for file_path in all_tif_files:
        parts = os.path.basename(file_path).split("_")
        if len(parts) < 3 or parts[0] != province:
            continue
        date_part = parts[1]
        if date_part.startswith(target_prefix):
            dates.append(date_part)
    return sorted(set(dates))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting MODIS county-stat extraction")
    print(f"Province/file prefix: {args.province}")
    print(f"MODIS folder: {args.modis_folder}")
    print(f"County boundary: {args.county}")
    print(f"Output folder: {args.output_dir}")
    print(f"Years: {args.start_year}-{args.end_year}")

    all_tif_files = glob.glob(os.path.join(args.modis_folder, "*.tif"))
    if not all_tif_files:
        print("Error: MODIS folder contains no tif files.")
        return

    for year in range(args.start_year, args.end_year + 1):
        print(f"\n{'=' * 60}")
        print(f"Processing {args.province} {year}")
        print(f"{'=' * 60}")

        dates = list_year_dates(all_tif_files, year, args.province)
        if not dates:
            print(f"Warning: no {year} MODIS tif files found for prefix {args.province}_A{year}")
            continue
        print(f"   -> Found {len(dates)} dates, first: {dates[0]}")

        corn_file = find_corn_file(year, args.ccd_maize_root, args.province)
        if not corn_file:
            print(f"Warning: skipping {year}, missing maize mask.")
            continue

        template_path = os.path.join(args.modis_folder, f"{args.province}_{dates[0]}_{INDICATORS[0]}.tif")
        if args.dry_run:
            print(f"   -> Template: {template_path}")
            print(f"   -> Maize mask: {corn_file}")
            continue

        try:
            valid_indices, valid_county_ids, id_to_name, id_to_code = get_valid_indices_and_ids(
                template_path, args.county, corn_file, args.name_column, args.code_column
            )
        except Exception as exc:
            print(f"Error: failed to build mask for {year}: {exc}")
            continue

        results = []
        print(f"Running {len(dates)} dates with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one_date,
                    date,
                    args.modis_folder,
                    args.province,
                    valid_indices,
                    valid_county_ids,
                    id_to_name,
                    id_to_code,
                ): date
                for date in dates
            }

            for future in tqdm(as_completed(futures), total=len(dates), desc=f"{year} progress"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"Error: {futures[future]} failed: {exc}")

        if results:
            full_df = pd.concat(results, ignore_index=True)
            sort_cols = ["日期", "区县代码"] if "区县代码" in full_df.columns else ["日期", "县名"]
            full_df = full_df.sort_values(by=sort_cols)

            leading_cols = ["日期"]
            if "区县代码" in full_df.columns:
                leading_cols.append("区县代码")
            leading_cols.append("县名")
            cols = leading_cols + [
                col for col in full_df.columns if col not in leading_cols + ["temp_id"]
            ]
            full_df = full_df[cols]

            save_name = f"{args.province}_{year}_Stats_Full.csv"
            save_path = os.path.join(args.output_dir, save_name)
            full_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"Saved {save_name}")
        else:
            print(f"Warning: no results for {year}.")

    print("\nAll requested years finished.")


if __name__ == "__main__":
    main()
