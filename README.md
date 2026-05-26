# DSTFusion

This repository contains data processing, training, evaluation, and plotting code for the DSTFusion corn yield prediction workflow.

## Directory Layout

```text
.
|-- data/                  # Input datasets and GeoJSON boundaries
|-- models/                # Saved standalone model weights
|-- outputs/               # Training, evaluation, figures, and generated images
|-- scripts/
|   |-- analysis/          # Province-level transfer and baseline analysis
|   |-- plotting/          # Publication and diagnostic plots
|   `-- utilities/         # One-off data/image conversion helpers
|-- data.py                # Dataset loading and feature construction
|-- ddcn_model.py          # DDCN model definitions
|-- eval_checkpoint.py     # Checkpoint evaluation entrypoint
`-- train.py               # Main training entrypoint
```

## Core Data

- `data/All_Data.csv`: daily county-level feature table.
- `data/Yield_Data.xlsx`: county-level yield targets.
- `data/SIF_Weekly.csv`: weekly SIF feature table.
- `data/jilin_phenology_2001-2020.csv`: phenology/static features.
- `data/Jilin_Soil_Texture_Combined_Static.csv`: soil texture features.
- `data/吉林省.json`: Jilin county boundary GeoJSON.
- `data/吉林省单产2000-2023.xlsx`: province-level yield targets.

## Training

Run from the project root:

```bash
python train.py
```

Important defaults now point to `data/`:

```bash
python train.py ^
  --train data/All_Data.csv ^
  --yield-csv data/Yield_Data.xlsx ^
  --weekly-csv data/SIF_Weekly.csv ^
  --static-csv data/jilin_phenology_2001-2020.csv ^
  --soil-csv data/Jilin_Soil_Texture_Combined_Static.csv ^
  --geojson data/吉林省.json ^
  --out-dir outputs
```

## Evaluation

```bash
python eval_checkpoint.py ^
  --checkpoint outputs/dual_tower/model.pt ^
  --data data/All_Data.csv ^
  --yield-csv data/Yield_Data.xlsx ^
  --static-csv data/jilin_phenology_2001-2020.csv
```

## Plotting And Analysis

Run plotting and analysis scripts from the project root so their default `outputs/` paths resolve correctly:

```bash
python scripts/plotting/plot_results.py
python scripts/plotting/plot_scatter_density_models.py
python scripts/plotting/plot_choropleth_models.py --geojson data/吉林省.json
python scripts/analysis/province_from_county_transfer.py
python scripts/analysis/compare_province_baselines.py
```

## Model Summary

- Week Transformer: 7 daily steps by feature columns into a weekly representation.
- Stage AT-LSTM: vegetative and reproductive stages.
- Season AT-LSTM: aggregates stage features to predict yield.

Common model options:

```bash
python train.py --stage-ratio 10,11 --stage-hidden 256 --season-hidden 256
```
