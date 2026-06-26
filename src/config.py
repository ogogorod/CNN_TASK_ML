"""
Central configuration for the weapon-classification pipeline.

Edit paths / hyperparameters here rather than scattering magic numbers
through the codebase.

Expected project layout:

weapon_classifier/
    data/
        raw/
            img/
            ann/
        processed/
    outputs/
        checkpoints/
        reports/
    src/
        config.py
        ...
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# SSL certificates
# ---------------------------------------------------------------------------
# Helps torchvision download pretrained model weights if needed.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# config.py is inside src/
# parents[1] means: go one level up from src/ to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Expected dataset layout:
#   data/raw/img/<file>.jpg
#   data/raw/ann/<file>.jpg.json
#
# You can still override these paths using environment variables:
#   WEAPON_DATA_ROOT
#   WEAPON_IMG_DIR
#   WEAPON_ANN_DIR
DATA_ROOT = Path(
    os.environ.get(
        "WEAPON_DATA_ROOT",
        PROJECT_ROOT / "data" / "raw",
    )
)

IMG_DIR = Path(
    os.environ.get(
        "WEAPON_IMG_DIR",
        DATA_ROOT / "img",
    )
)

ANN_DIR = Path(
    os.environ.get(
        "WEAPON_ANN_DIR",
        DATA_ROOT / "ann",
    )
)

# Outputs: checkpoints, reports, plots, metrics
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"

# Processed datasets, for example exported balanced test split
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Convert Path objects to strings for compatibility with os.path, glob, etc.
PROJECT_ROOT = str(PROJECT_ROOT)
DATA_ROOT = str(DATA_ROOT)
IMG_DIR = str(IMG_DIR)
ANN_DIR = str(ANN_DIR)
OUTPUT_DIR = str(OUTPUT_DIR)
CHECKPOINT_DIR = str(CHECKPOINT_DIR)
REPORT_DIR = str(REPORT_DIR)
PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Alphabetical, fixed order -> stable label indices across runs.
CLASS_NAMES = ["billete", "knife", "monedero", "pistol", "smartphone", "tarjeta"]

CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_IDX.items()}

NUM_CLASSES = len(CLASS_NAMES)

# Which classes count as "weapon" for the derived binary task.
WEAPON_CLASSES = {"knife", "pistol"}
WEAPON_IDXS = {CLASS_TO_IDX[class_name] for class_name in WEAPON_CLASSES}


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
IMG_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 4

EPOCHS_SCRATCH = 30
EPOCHS_TRANSFER = 15

LR_SCRATCH = 1e-3
LR_TRANSFER = 1e-4

WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6

# utils.get_device() still decides automatically:
# cuda -> mps -> cpu
DEVICE = "cuda"
