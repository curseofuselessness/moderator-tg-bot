"""Paths to project folders."""
from pathlib import Path

# shared/config/paths.py -> shared/config -> shared -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"