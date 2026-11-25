# DST-Fusion 论文图生成脚本

图号以 2026-11-25 的 Remote Sensing 稿件 `paper.tex` 为准，涵盖正文图 1、3–5 和附录图 A1–A5。正文图 3、图 5 的子图分别生成，现稿的排版、标号和图注见 `layout/figure_blocks.tex`。这些图对应的原 PDF 在 `reference_pdfs/current/`。上次编译的 PDF 中保留的旧版误差图和消融图也随包提供。

## 图号、脚本和输入

| 图号 | 稿件文件 | 生成脚本及数据 |
| --- | --- | --- |
| 图 1 研究区 | `ren1.pdf` | `scripts/plot_submission_study_area.py`。吉林县界、33 个建模县的名单和 2001–2019 年 CCD-Maize 显示图层。 |
| 图 3(a–b) 五年平均指标 | `ren3.pdf`、`ren4.pdf` | `scripts/plot_results.py`。六模型原版年度指标。先按模型和年份汇总，再对 2015–2019 年等权求均值与年度总体标准差。 |
| 图 3(c) 密度散点 | `ren5.pdf` | `scripts/plot_scatter_density_models.py`。六模型各 155 个原版县年预测与观测。散点合并五年，图内 RMSE 和 R² 标签来自等权年度指标。 |
| 图 4 六模型县级误差 | `county_error_maps/figure4.pdf` | `scripts/county_error_maps/analyze_and_plot.py`。使用图 4 归档的 seed 42 六模型县年预测和同一份吉林县界，按县的实际评价年份计算有符号均值和 MAE。脚本同时导出县年误差、县均值和 Queen 权重下的 Moran’s I 结果。 |
| 图 5(a–b) 年度指标 | `ren7.pdf`、`ren8.pdf` | `scripts/plot_results.py`。与图 3(a–b) 共用六模型原版年度指标。 |
| 图 A1–A5 年度县级误差 | `county_error_maps/annual_maps.pdf` 第 1–5 页 | 与图 4 共用脚本和预测。2015–2019 年县数依次为 33、33、28、28、33。 |

图 3 和图 5 使用 `inputs/code/outputs/` 下六模型的原版实验归档，图 3(a) 的 DST-Fusion 平均 RMSE 显示为 **0.701**。图 4 和附录地图按其原图的预测清单生成，其中 DST-Fusion、DDCN 用服务器复跑的县年预测，其余四个模型用归档原版预测。清单保留预测文件校验值，脚本运行时会核对。两处 Transformer-daily 均取原版基线目录。

## 运行

在本文件夹内，使用 Python 3.12 安装依赖并生成现稿各图。运行无需联网获取绘图数据。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_figures.py --output-dir rebuilt
```

可以只跑指定图号。图 4 和附录由同一程序生成，选中任一种会同时输出两份地图。

```sh
python scripts/build_figures.py --output-dir rebuilt --figures 3 5
python scripts/build_figures.py --output-dir rebuilt --figures 4 appendix
```

默认使用包内图 1 显示图层。以下命令可从原始年度分类图重建该图层。脚本将每张原始分类图以最近邻缩至长宽的十分之一，像元在 19 张图中至少出现 6 次且位于吉林省内才绘制。输入目录结构为 `YYYY/classified-Jilin-maize-YYYY-WGS84-v1.tif`。[CCD-Maize 数据产品](https://www.scidb.cn/en/detail?dataSetId=111aee7914884f59bbdd077d7b2e926b)

```sh
python scripts/build_figures.py --output-dir rebuilt --figures 1 \
  --ccd-maize /path/to/CCD-Maize
```

## 上次编译稿的图

`--figures prior` 重绘旧版图 4 的单组有符号误差图 `ren6.pdf`，以及当时图 6 的消融平均 RMSE 和年度 RMSE `ren9.pdf`、`ren10.pdf`。原导出件在 `reference_pdfs/prior/`。

```sh
python scripts/build_figures.py --output-dir rebuilt --figures prior
```

旧版 `ren6` 按其当时使用的预测文件着色，并读取旧版指标注记所用的结果表。脚本重绘了地图元素，原 PDF 中的后期排版以随包参考件为准。

## 核对文件

- `layout/figure_blocks.tex` 收录本包覆盖的九个图环境，并沿用原稿图号。需要完整编译时，原稿仍负责文档类和其他宏。
- `provenance/verification.json` 记录本次重跑、原图页数、数据口径和核对结果。
- `provenance/county_error_maps_original.json` 和 `scripts/figures_replot/provenance.json` 记录当前地图的原预测来源。Queen 边表在 `inputs/queen_edges.csv`。
- 可运行 `shasum -a 256 -c SHA256SUMS.txt` 核对包内文件。
