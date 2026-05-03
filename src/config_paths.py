from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
SEGMENTATION_RESULTS_DIR = RESULTS_DIR / "segmentation"
FEATURE_SIGNIFICANCE_DIR = RESULTS_DIR / "feature_significance_analysis"
FEATURE_SET_COMPARISON_DIR = RESULTS_DIR / "feature_set_comparison"
FEATURE_PRUNING_DIR = RESULTS_DIR / "feature_pruning_analysis"
COMPACT_MODEL_DIR = RESULTS_DIR / "compact_healing_model"
COMPACT_PREDICTIONS_DIR = RESULTS_DIR / "compact_healing_predictions"
COMPACT_REPORT_DIR = RESULTS_DIR / "compact_healing_report"

PRACTICE_3P_DIR = PROJECT_ROOT / "practice_3p"
PRACTICE_3P_FIGURES_DIR = PRACTICE_3P_DIR / "figures"

WOUND_IMAGE_FEATURES_CSV = INTERIM_DATA_DIR / "wound_image_features.csv"
WOUND_PHASE_DYNAMICS_CSV = INTERIM_DATA_DIR / "wound_phase_dynamics.csv"
WOUND_FORECAST_DATASET_CSV = PROCESSED_DATA_DIR / "wound_forecast_dataset.csv"
SEGMENTATION_WARNINGS_LOG = SEGMENTATION_RESULTS_DIR / "segmentation_warnings.log"


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        SEGMENTATION_RESULTS_DIR,
        FEATURE_SIGNIFICANCE_DIR,
        FEATURE_SET_COMPARISON_DIR,
        FEATURE_PRUNING_DIR,
        COMPACT_MODEL_DIR,
        COMPACT_PREDICTIONS_DIR,
        COMPACT_REPORT_DIR,
        PRACTICE_3P_DIR,
        PRACTICE_3P_FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
