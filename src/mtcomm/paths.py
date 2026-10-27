"""Locations of data and outputs, which live outside the git repo."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent

DATA_ROOT = Path(os.environ.get("MTCOMM_DATA", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("MTCOMM_OUTPUTS", PROJECT_ROOT / "outputs" / "mtcomm"))
ENV_FILE = Path(os.environ.get("MTCOMM_ENV", PROJECT_ROOT / ".env"))

XM3600_DIR = DATA_ROOT / "external" / "xm3600"
BLEND_DIR = DATA_ROOT / "external" / "blend"
ANNOTATION_DIR = DATA_ROOT / "annotation"
