# DST-Fusion 上游预处理

本包包含吉林 2001–2019 年的历史脚本、县级中间表、实际训练输入，以及从 CCD-Maize 原始分类图重建的 2015–2019 年五折玉米掩膜。各项命令均从本文件夹运行。原始 NetCDF、HDF 和全国 HCSIF GeoTIFF 由下列产品入口获取。

## 原始产品与处理后输入

| 流程 | 原始来源与本地文件形式 | 包内产物 |
| --- | --- | --- |
| 逐折玉米掩膜 | Peng 等的 CCD-Maize。2001–2020 年年度 30 m 玉米分类，WGS84，0 为非玉米，1 为玉米。本项目吉林文件为 `YYYY/classified-Jilin-maize-YYYY-WGS84-v1.tif`。[Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=111aee7914884f59bbdd077d7b2e926b) | `intermediate/fold_masks_30m/` 有五个重建的 30% GeoTIFF。 |
| 县级气象 | 中国区域地面气象驱动数据集 CMFD v2.0 的 0.1° 逐日 `temp`、`prec`、`pres`、`shum`。脚本另读 `tmax/YYYY.nc`，其上游产品和日最高温生成记录未找到。[国家青藏高原科学数据中心](https://doi.org/10.11888/Atmos.tpdc.302088) | `intermediate/cmfd/` 有 19 张县日表。六个气象列并入 `inputs/model/All_Data.csv`。 |
| 植被指数 | NASA LP DAAC Terra MODIS `MOD09GA.061`，500 m 反射率和 1 km 状态 QA。使用吉林对应的 `h26v04`、`h27v04` 瓦片，文件为 `MOD09GA.AYYYYDDD.hXXvYY.061.*.hdf`。[NASA Earthdata](https://access.earthdata.nasa.gov/citations/CIT3804247588-ESDIS) · [Earth Engine 对应产品](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA) | `intermediate/modis_before_fill/` 和 `intermediate/modis_after_fill/` 各有 19 张县日表，包含五指数的中位数、标准差、P10 和 P90。平滑后的指数列用于 `inputs/model/All_Data.csv`。 |
| HCSIF 周序列 | Tao 等的中国高分辨率卫星 SIF 产品 HCSIF，2000–2022 年，500 m、8 天。脚本读取 `YYYY/YYYYDDD.tif`。[原始产品 DOI](https://doi.org/10.57760/sciencedb.16910) · [产品记录](https://www.nesdc.org.cn/sdo/detail?id=674996897e281765a35650cf&subjectCode=1704) · [产品方法](https://doi.org/10.1038/s41597-024-04101-6) | `intermediate/hcsif_8day/` 有 19 张县级 8 天表。加权周化后得到 `inputs/model/SIF_Weekly.csv`。 |

共用县界是 `inputs/boundaries/吉林省.json`，沿用训练归档中的 60 个县级单元，字段 `name` 和 `adcode` 分别表示名称和行政区划代码。归档未标明此 GeoJSON 的最初发布机构和版次。

| 实际训练输入 | 主键与规模 | 模型使用字段 |
| --- | --- | --- |
| `inputs/model/All_Data.csv` | `Date, County`，129,064 行，源表含 34 个县名 | 五指数各取 `median`、`p10`、`p90`，外加 GDD、KDD、PRCP、VPD，共 19 个逐日特征。标准差、县均温和县均最高温保留在表中。 |
| `inputs/model/SIF_Weekly.csv` | `year, name, week_index`，28,500 行，60 个县级单元 | `SIF_median` 和 `sif_missing_flag`。表中也保留周起止、重叠天数和其余 SIF 统计量。 |

两张训练表和县界的 SHA-256 与归档运行 `20261102_1543_box` 一致。表中县名数是输入文件的覆盖范围，评估县按训练归档的样本选择另定。两张训练表均无折次字段。历史 MODIS 和 HCSIF 提取脚本默认载入观测年的 CCD-Maize 图，尚缺五折主实验的逐折提取记录来判定实际训练表是否来自 30% 折掩膜。本包提供独立的固定折掩膜提取命令和由此新算的 MODIS 示例。归档 MODIS 县日表没有有效像元数或掩膜候选像元数，无法仅凭统计值恢复它们。

## 环境

建议使用 Python 3.12。周化脚本仅用标准库。运行其余流程可安装包内依赖。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

下列命令中的 `/path/to/...` 替换为下载后的数据目录。重跑输出统一写入 `work/`，包内训练文件可直接作为归档输入使用。

## 1. 重建逐折玉米掩膜

目标年为 `t` 时，固定用 2001 至 `t−1` 年的吉林玉米图。像元历史玉米出现次数达到 `ceil(0.30 × 历史年数)` 即保留。2015–2019 年五折的历史年数依次是 14、15、16、17、18，最少出现次数是 5、5、5、6、6。目标年分类图仅用于事后重叠核验。当折内重复提取历史年份与目标年影像时，均使用该折同一张掩膜。30% 指历史出现频率。

```sh
python scripts/01_maize/build_fold_masks.py \
  --ccd-root /path/to/CCD-Maize --output-dir work/fold_masks_30m
```

输出 GeoTIFF 在 CCD-Maize 原网格上保存 0/1，标签记录目标年、历史起止年和阈值。包内五张掩膜已与 `provenance/county_fold_overlap.csv` 中 165 条 30% 折次–县记录核对，县内像元数全部一致。核对采用原 30 m 审查脚本的县界栅格化规则。

`scripts/01_maize/analyze_overlap.py` 是历史重叠审查脚本，已调整县界和评估键的读取位置。它另外需要 2015–2019 年目标分类图来核对重叠。包内两张重叠表来自原审查。复算命令如下。

```sh
python scripts/01_maize/analyze_overlap.py /path/to/CCD-Maize \
  --output work/mask_overlap --historical-manifest provenance/mask_sensitivity_manifest.json
```

`intermediate/mask_sensitivity/` 另存历史敏感性实验的 MODIS 网格数据。五波段计数文件依次对应 2015–2019 年，按原清单的格网将逐年 CCD-Maize 图先最近邻采样，再累计出现次数。同目录 20% 和 50% 栅格为该实验的二值掩膜。以上与 `fold_masks_30m/` 的构造顺序、格网各有记录。

## 2. CMFD 县级气象聚合

使用 `scripts/02_cmfd/climate_pre.py`。把脚本复制为 `work/climate_pre.py`，设置 `cmfd_dir`、`tmax_dir`、`json_path` 和 `output_dir`，按需调整 `years_to_process`。CMFD 目录内每年需要四个文件，统一命名为 `<变量>_CMFD_V0200_B-01_01dy_010deg_YYYY01-YYYY12.nc`。其中 `temp` 单位 K、`pres` 单位 Pa、`shum` 单位 kg/kg，`prec` 是降水率。独立 Tmax 文件需含变量 `tmax`，历史脚本按 °C 直接读取。执行工作副本 `python work/climate_pre.py` 即导出 `吉林_气候数据_YYYY.csv`。

1. 原代码通过 `regionmask.Regions(...).mask(lon, lat)`，以**格点中心归属**匹配到**完整行政县界**，精确落在边界的点由 regionmask 的边界规则处理。入选格点等权，缺值跳过。气象处理覆盖县内玉米区以外的格点。[regionmask 格点和边界规则](https://regionmask.readthedocs.io/en/latest/notebooks/method.html)
2. 分别对原始变量求县日均值，Tmax 在自己的网格上同样处理，然后按县与日期合并。县均温先由 K 转 °C，随后求 `GDD = max(县均温 − 10, 0)` 和 `KDD = max(县均 Tmax − 29, 0)`。VPD 用县均温、县均压和县均比湿计算，负值置零，单位 kPa。降水率乘 86,400 转为每日降水量，`PRCP` 以 mm/day 计。
3. 19 张归档表共有 416,340 个县日。与实际 `All_Data.csv` 的 129,064 个县日逐键核对，六个气象列最大绝对差为 `5.0e-8`。

## 3. MODIS 植被指数

**HDF 到指数栅格。** `batch_modis_pre.py` 是历史批处理脚本。可编辑工作副本的 `INPUT_FOLDER` 和 `OUTPUT_FOLDER` 批量运行，输入目录需直接含两瓦片 HDF，GDAL 需支持 HDF4。也可以使用包内 `hdf_to_indices.py` 逐日调用同一指数算法，由 PyHDF 读取 HDF4 子数据集。下面的日期和瓦片已用原始 HDF 实跑。

```sh
python scripts/03_modis/hdf_to_indices.py \
  --hdf-root /path/to/MOD09GA --date 2015-06-01 --output-dir work/modis_indices
```

算法读取 b01 红、b02 近红外、b03 蓝、b04 绿、b06 短波红外和 `state_1km`。拼接瓦片，将 1 km QA 最近邻对齐到反射率网格，只保留云状态为 0、无云影且 QA 有值的像元。算出 EVI、WDRVI、GCI、NDWI、NIRv 后，用双线性重投影到 EPSG:4326 的 0.005° 网格。历史代码中的反射率 `×0.0001` 是注释，包内调用保持原数值口径。产品本身的标度系数为 0.0001，转换成物理反射率会影响至少 EVI 和 NIRv。

**县级提取。** `modis_indices.py` 用 `all_touched=True` 栅格化县界，并把玉米掩膜最近邻对齐至指数网格。原脚本可显式传入 `--modis-folder`、`--ccd-maize-root`、`--county`、`--output-dir`、`--start-year`、`--end-year`，按观测年加载年度玉米图。逐折分析使用新增的 `extract_fold_daily.py`，为每折固定一张掩膜。

```sh
python scripts/03_modis/extract_fold_daily.py \
  --indices work/modis_indices --county inputs/boundaries/吉林省.json \
  --mask intermediate/fold_masks_30m/Jilin_fold_2015_freq30_30m.tif \
  --fold 2015 --start-date 2015-06-01 --end-date 2015-06-01 \
  --output work/modis_fold_2015_before_fill.csv
```

输出每县、日、折的中位数、标准差、P10/P90 和各指数有效像元数。有效比例等于相应指数的有限值像元数除以县内掩膜候选像元数。候选数为零时比例留空。指数栅格缺失时，该日该指数的统计值留空，有效像元数为零。包内 `intermediate/examples/modis_fold_2015_2015-06-01_before_fill.csv` 是此命令的实测示例。

**插值、平滑和合并。** 历史县日表各县内按已有记录日期排序，双向线性插值、首尾用最近观测补齐，再用二阶、15 点 Savitzky–Golay 滤波。训练读取时再对齐到 DOY 128–302。包含像元计数的提取文件需单独保留。若要对新逐折表应用历史平滑脚本，先按折及观测年导出县名、日期、县代码和 20 个指数统计列，去掉折次、候选数、有效数及比例，日期转成 `AYYYYDDD`。

```sh
python scripts/03_modis/modis_pre.py \
  --input-dir intermediate/modis_before_fill --output-dir work/modis_after_fill
```

历史 `modis_climate_join.py` 按 `Date, County` 对平滑指数表和气象表做内连接。包内 `join_daily.py` 采用同一连接键，逐文件识别指数表的 UTF-8 或 GB18030 编码。用以下命令合并 60 个县级单元的归档表。

```sh
python scripts/03_modis/join_daily.py \
  --veg-dir intermediate/modis_after_fill --climate-dir intermediate/cmfd \
  --output work/modis_cmfd_county_daily.csv
```

19 年归档补前表用现有脚本重跑后，平滑结果与归档补后表逐键核对，4,555,200 个统计值中有 36 个相差大于 `1e-6`，最大绝对差为 `4.22e-4`。将归档补后表再与实际 `All_Data.csv` 按训练键核对，模型使用的 1,935,960 个指数值中，1,935,884 个在 `1e-6` 内一致（含两边均缺失）。包内保留两份归档表供核对。

## 4. HCSIF 县级提取与周化

`tif_to_sequence.py` 是历史栅格提取脚本。它按吉林县界外包框裁切全国 GeoTIFF，最近邻对齐玉米掩膜，用 `all_touched=True` 栅格化县界，对原始有效值乘 0.0001，输出各县中位数、标准差、P10 和 P90。包内 `extract_hcsif.py` 负责路径和掩膜选择。按观测年用年度玉米图时传入 `--ccd-root`。为一折内的所有年份选用同一掩膜时，传入 `--fold-mask`。

```sh
python scripts/04_hcsif/extract_hcsif.py \
  --sif-root /path/to/HCSIF --county inputs/boundaries/吉林省.json \
  --fold-mask intermediate/fold_masks_30m/Jilin_fold_2015_freq30_30m.tif \
  --start-year 2001 --end-year 2015 --output-dir work/hcsif_fold_2015_8day
```

以下从随包提供的历史 8 天表重算周序列。如需为某折生成周序列，将输入目录换为该折的 8 天表目录。

```sh
python scripts/04_hcsif/sif_8day_to_week.py \
  --input-dir intermediate/hcsif_8day --output-dir work/hcsif_weekly
```

每年从 DOY 128 到 302 切成 25 个连续的 7 天周。对每个 SIF 统计量，以有效 8 天合成期与周的重叠天数为权重计算均值。有效重叠少于 3 天时该值缺失。任一统计量达不到 3 天时，共用的 `sif_missing_flag` 为 1。这里的 `sif_overlap_days` 为时间区间的总重叠天数，可能包含该统计量缺失的合成期。将年度周表合并、把 `县名` 改为 `name` 后，与随包训练周表按键核对，28,500 行的周界、缺失标记、重叠天数和四个 SIF 数值全一致。

## 文件核对

- `provenance/model_input_archive.json` 记录训练输入的归档 SHA-256。
- `provenance/overlap_provenance.json`、`mask_sensitivity_manifest.json` 记录 CCD-Maize 源图版本、逐年校验值与格网。
- `provenance/modis_reader_pilot.json` 是既有的 HDF 与 Earth Engine 读数对照，保留与旧县日统计的逐项比较。
- `provenance/verification.json` 记录本次核对范围。可在支持 `shasum` 的系统运行 `shasum -a 256 -c SHA256SUMS.txt` 检查包内文件。
