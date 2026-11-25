"""Compare pre-target CCD-Maize masks with the target crop map on the source grid."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely import make_valid
from shapely.geometry import shape


YEARS = range(2001, 2020)
TARGET_YEARS = range(2015, 2020)
THRESHOLDS = (20, 30, 50)
COUNTY_NAMES = {
    "安图县": "Antu County", "长白朝鲜族自治县": "Changbai Korean Autonomous County",
    "长岭县": "Changling County", "大安市": "Da'an City", "敦化市": "Dunhua City",
    "抚松县": "Fusong County", "公主岭市": "Gongzhuling City", "和龙市": "Helong City",
    "桦甸市": "Huadian City", "辉南县": "Huinan County", "珲春市": "Hunchun City",
    "集安市": "Ji'an City", "蛟河市": "Jiaohe City", "靖宇县": "Jingyu County",
    "临江市": "Linjiang City", "梨树县": "Lishu County", "柳河县": "Liuhe County",
    "龙井市": "Longjing City", "梅河口市": "Meihekou City", "磐石市": "Panshi City",
    "前郭尔罗斯蒙古族自治县": "Qian Gorlos Mongolian Autonomous County",
    "乾安县": "Qian'an County", "双辽市": "Shuangliao City", "舒兰市": "Shulan City",
    "洮南市": "Taonan City", "通化县": "Tonghua County", "通榆县": "Tongyu County",
    "图们市": "Tumen City", "汪清县": "Wangqing County", "延吉市": "Yanji City",
    "伊通满族自治县": "Yitong Manchu Autonomous County", "永吉县": "Yongji County",
    "镇赉县": "Zhenlai County",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def area_per_row(transform, height):
    """Compute WGS84 ellipsoidal cell areas in hectares, one value per latitude row."""
    geod = Geod(ellps="WGS84")
    longitudes = [transform.c, transform.c + transform.a, transform.c + transform.a, transform.c]
    latitudes = transform.f + np.arange(height + 1) * transform.e
    return np.array([
        abs(geod.polygon_area_perimeter(longitudes, [north, north, south, south])[0]) / 10000
        for north, south in zip(latitudes[:-1], latitudes[1:])
    ])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ccd_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    boundary_path = project / "inputs/boundaries/吉林省.json"
    eval_path = project / "provenance/evaluation_county_year_keys.csv"
    with eval_path.open(encoding="utf-8-sig", newline="") as stream:
        records = [(row["county"], int(row["val_year"])) for row in csv.DictReader(stream)]
    eval_pairs = set(records)
    assert len(eval_pairs) == len(records)
    assert Counter(year for _, year in eval_pairs) == dict(zip(TARGET_YEARS, [33, 33, 28, 28, 33]))
    assert {county for county, _ in eval_pairs} == set(COUNTY_NAMES)

    features = json.loads(boundary_path.read_text(encoding="utf-8"))["features"]
    counties = sorted(features, key=lambda feature: feature["properties"]["adcode"])
    county_codes = {feature["properties"]["name"]: i for i, feature in enumerate(counties, 1)}
    assert len(county_codes) == len(counties) == 60 and set(COUNTY_NAMES) <= county_codes.keys()
    shapes = []
    repaired = []
    for i, feature in enumerate(counties, 1):
        geometry = shape(feature["geometry"])
        if not geometry.is_valid:
            repaired.append(feature["properties"]["name"])
            geometry = make_valid(geometry, method="structure", keep_collapsed=False)
        assert geometry.geom_type in ("Polygon", "MultiPolygon") and not geometry.is_empty
        shapes.append((geometry, i))

    paths = {year: args.ccd_root / str(year) / f"classified-Jilin-maize-{year}-WGS84-v1.tif" for year in YEARS}
    checksums = {year: sha256(path) for year, path in paths.items()}
    if args.historical_manifest:
        source_manifest = json.loads(args.historical_manifest.read_text())
        assert [record["year"] for record in source_manifest["sources"]] == list(range(2001, 2019))
        assert all(checksums[record["year"]] == record["sha256"] for record in source_manifest["sources"])
    print(f"Validated source files for {len(YEARS)} years", flush=True)

    # Index each county's cells by historical occurrence count and target-year class.
    group_size = len(counties) + 1
    hist_shape = (len(YEARS), 2, group_size)
    counts = np.zeros((len(TARGET_YEARS), *hist_shape), dtype=np.int64)
    areas = np.zeros_like(counts, dtype=np.float64)
    province_pixels = np.zeros(len(TARGET_YEARS), dtype=np.int64)
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), ExitStack() as stack:
        maps = {year: stack.enter_context(rasterio.open(path)) for year, path in paths.items()}
        first = maps[2001]
        grid = (first.shape, first.transform, first.crs)
        assert first.crs.to_epsg() == 4326 and first.transform.is_rectilinear
        assert first.transform.a > 0 and first.transform.e < 0 and first.transform.b == first.transform.d == 0
        assert all((r.shape, r.transform, r.crs) == grid and r.nodata is None and r.count == 1
                   and r.dtypes == ("uint8",) for r in maps.values())
        cell_ha = area_per_row(first.transform, first.height)
        overlap_boundary_cells = 0
        for top in range(0, first.height, 192):
            height = min(192, first.height - top)
            window = Window(0, top, first.width, height)
            transform = first.window_transform(window)
            window_bottom = transform.f + height * transform.e
            window_shapes = [(geom, code) for geom, code in shapes
                             if geom.bounds[3] >= window_bottom and geom.bounds[1] <= transform.f]
            labels = rasterize(window_shapes, out_shape=(height, first.width), transform=transform,
                               dtype="uint8", all_touched=False) if window_shapes else np.zeros((height, first.width), dtype=np.uint8)
            # Reverse the burn order to detect any overlapping county pixel assignments.
            if window_shapes:
                reverse = rasterize(window_shapes[::-1], out_shape=labels.shape, transform=transform,
                                    dtype="uint8", all_touched=False)
                overlap_boundary_cells += int(np.count_nonzero(labels != reverse))
            yy, xx = np.nonzero(labels)
            zones = labels[yy, xx].astype(np.int32)
            weights = cell_ha[top + yy]
            history = np.zeros(len(zones), dtype=np.uint8)
            for year, raster in maps.items():
                pixels = raster.read(1, window=window)
                assert pixels.max() <= 1, ("Unexpected CCD-Maize class", year, top)
                current = pixels[yy, xx]
                if year in TARGET_YEARS:
                    fold = year - TARGET_YEARS.start
                    province_pixels[fold] += np.count_nonzero(pixels)
                    codes = (2 * history.astype(np.int32) + current) * group_size + zones
                    counts[fold] += np.bincount(codes, minlength=np.prod(hist_shape)).reshape(hist_shape)
                    areas[fold] += np.bincount(codes, weights=weights, minlength=np.prod(hist_shape)).reshape(hist_shape)
                history += current
            if top % (192 * 20) == 0:
                print(f"Processed raster rows {top + height:,}/{first.height:,}", flush=True)

    # Every fold covers the same county pixels. Annual maize areas must be a subset of that grid.
    assert np.all(counts.sum(axis=(1, 2)) == counts[0].sum(axis=(0, 1)))
    assert np.all(counts[:, :, 1, :].sum(axis=(1, 2)) <= province_pixels)
    assert np.all(counts.sum(axis=(1, 2))[:, 1:] > 0)
    assert np.allclose(areas.sum(axis=(1, 2)), areas[0].sum(axis=(0, 1)))

    detail = []
    summary = []
    names = sorted(COUNTY_NAMES, key=COUNTY_NAMES.get)
    for fold, year in enumerate(TARGET_YEARS):
        total_years = year - YEARS.start
        assert counts[fold, total_years + 1:].sum() == 0
        last_mask = None
        for threshold in THRESHOLDS:
            minimum = (threshold * total_years + 99) // 100
            count_cube, area_cube = counts[fold], areas[fold]
            mask_count, maize_count, intersect_count = (count_cube[minimum:].sum(axis=(0, 1)),
                                                       count_cube[:, 1].sum(axis=0),
                                                       count_cube[minimum:, 1].sum(axis=0))
            mask_area, maize_area, intersect_area = (area_cube[minimum:].sum(axis=(0, 1)),
                                                    area_cube[:, 1].sum(axis=0),
                                                    area_cube[minimum:, 1].sum(axis=0))
            assert np.all(intersect_count <= np.minimum(mask_count, maize_count))
            if last_mask is not None:
                assert np.all(mask_count <= last_mask)
            last_mask = mask_count

            rows = []
            for name in names:
                code = county_codes[name]
                mask, maize, overlap = float(mask_area[code]), float(maize_area[code]), float(intersect_area[code])
                rows.append(dict(
                    county=COUNTY_NAMES[name], county_code=int(counties[code - 1]["properties"]["adcode"]),
                    target_year=year, threshold=threshold / 100, historical_start=YEARS.start,
                    historical_end=year - 1, historical_years=total_years, min_maize_years=minimum,
                    evaluated_yield=(name, year) in eval_pairs, mask_pixels=int(mask_count[code]),
                    mask_ha=mask, target_maize_pixels=int(maize_count[code]), target_maize_ha=maize,
                    overlap_pixels=int(intersect_count[code]), overlap_ha=overlap, area_difference_ha=mask-maize,
                    maize_share_of_mask=overlap/mask if mask else None,
                    covered_share_of_target_maize=overlap/maize if maize else None,
                ))
            detail.extend(rows)
            mask_ha = sum(row["mask_ha"] for row in rows)
            maize_ha = sum(row["target_maize_ha"] for row in rows)
            overlap_ha = sum(row["overlap_ha"] for row in rows)
            summary.append(dict(
                target_year=year, threshold=threshold / 100, min_maize_years=minimum, counties=len(rows),
                evaluated_counties=sum(row["evaluated_yield"] for row in rows),
                mask_pixels=sum(row["mask_pixels"] for row in rows), mask_ha=mask_ha,
                target_maize_ha=maize_ha, overlap_ha=overlap_ha, area_difference_ha=mask_ha-maize_ha,
                maize_share_of_mask=overlap_ha/mask_ha, covered_share_of_target_maize=overlap_ha/maize_ha,
                median_county_maize_share_of_mask=float(np.median([r["maize_share_of_mask"] for r in rows if r["mask_ha"]])),
                median_county_covered_share=float(np.median([r["covered_share_of_target_maize"] for r in rows if r["target_maize_ha"]])),
            ))

    assert len(detail) == 495 and len(summary) == 15
    assert len({(r["county_code"], r["target_year"], r["threshold"]) for r in detail}) == 495
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "county_fold_overlap.csv", detail)
    write_csv(args.output / "annual_overlap.csv", summary)
    provenance = dict(
        product_doi="10.57760/SCIENCEDB.08490", version="WGS84-v1", epsg=4326, raster_shape=grid[0],
        transform=tuple(grid[1])[:6], cell_area_ha_range=[cell_ha.min(), cell_ha.max()],
        historical_manifest_verified=bool(args.historical_manifest),
        rasters=[dict(year=year, filename=paths[year].name, sha256=checksums[year]) for year in YEARS],
        boundary_file=str(boundary_path.relative_to(project)), boundary_sha256=sha256(boundary_path),
        repaired_counties=repaired, ambiguous_boundary_pixels=overlap_boundary_cells,
        evaluation_file=str(eval_path.relative_to(project)), evaluation_sha256=sha256(eval_path),
        assignment="Pixel-center rasterization, ascending administrative code burn order",
        area="WGS84 ellipsoidal pixel area by latitude row, converted to hectares",
        masks="Binary maps on their original shared grid. Each mask uses every annual map through target year minus one.",
    )
    (args.output / "overlap_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    (args.output / "overlap_tables.json").write_text(json.dumps(dict(detail=detail, summary=summary)) + "\n")
    print(f"Wrote {len(detail)} county-fold-threshold and {len(summary)} annual records", flush=True)


if __name__ == "__main__":
    main()
