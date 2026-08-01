from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ARTIFACTS = REPOSITORY_ROOT.parent / "acceleration_retrieval" / "artifacts_dataset_split"
DEFAULT_ARTIFACTS = REPOSITORY_ROOT / "artifacts"

HISTORY_MONTHS = 5
MIN_HISTORY_MONTHS = 3
FORECAST_MONTHS = 12
MIN_TARGET_MONTHS = 8
MIN_GUIDE_MONTHS = 8
EMBEDDING_DIM = 256
TOP_K = 3

