# Data Pipeline

## Overview

The project uses multiple data sources that are normalized to a unified format,
merged, cleaned.

## Sources

| Source | Path | Description |
|---|---|---|
| Downloaded datasets | `data/raw/` | HF, Kaggle |
| Manual labeling | `data/external/` | 500 examples |
| Processed | `data/processed/` | Clean data |

## Data format

All datasets in `data/processed/` are normalized to a unified format:

| Column | Type | Description |
|---|---|---|
| `text` | str | Message text |
| `label` | int | 0 = not toxic, 1 = toxic |


```mermaid
flowchart TD
    raw[data/raw/] --> clean
    ext[data/external/] --> clean

    clean[clean] --> all_clean
    all_clean[data/processed/] --> merge
    merge[load.py <br> merge data] --> eda
    eda[EDA]


```

## Steps

### 1. Loading
`training/dataset/load.py` — merge all clean data from `data/processed/`.

```python
from training.dataset.load import load_all_data
df = load_all_data()
```

### 2. EDA
`notebooks/01_eda.ipynb` — exploratory analysis: class balance, text length, examples.


## Notes

- **Test set** — held out in advance, not touched until final evaluation
- **Reproducibility** — `processed/` is always regenerated from `raw/` + `external/`
- **Privacy** — `data/external/` is anonymized before committing
