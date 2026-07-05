import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from src.core.config_loader import config
from src.infra.logger import setup_logger
import matplotlib.pyplot as plt

# --- BOOTSTRAP SEQUENCE ---
setup_logger(config.logging, Path(config.project_root))

def profile_time_series(df: pd.DataFrame) -> None:
    """
    Performs a diagnostic dump of the dataset to characterize the layout,
    types, and dimensions.
    """
    logger.info("System: Starting Diagnostic Profiling...")
    
    # 1. Print Dimensions
    print(f"\n--- DIMENSIONS ---")
    print(f"Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")

    # 2. Print Data Types and Null Counts
    print(f"\n--- DATA TYPES & NULLS ---")
    print(df.info())

    # 3. Print Sample Data
    print(f"\n--- DATA SAMPLE (First 5 Rows) ---")
    print(df.head())

    # 4. Check for Interval Consistency (using index)
    try:
        timestamp_index = df.index.to_series()
        diffs = timestamp_index.diff()
        seconds_diff = diffs.dt.total_seconds()
        valid_diffs = seconds_diff.iloc[1:].dropna()

        if not valid_diffs.empty:
            intervals = valid_diffs / 900
            anomalies = intervals[intervals != 1.0].count()
            logger.info(f"System: Interval consistency check passed. Anomalies found: {anomalies}")
        else:
            logger.error("System: Interval check failed. Timestamp index contains NaT values.")
    except Exception as e:
        logger.error(f"System: Interval check failed: {e}")


def run_eda(df: pd.DataFrame) -> None:
    reports_dir = Path(config.artifacts.reports_path)
    print("Resolved reports path:", reports_dir.resolve())

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Shape
    (reports_dir / "shape.txt").write_text(str(df.shape))

    # Columns
    (reports_dir / "columns.txt").write_text("\n".join(df.columns))

    # Dtypes
    df.dtypes.to_csv(reports_dir / "dtypes.csv")

    # Missing values
    df.isna().sum().to_csv(reports_dir / "missing_values.csv")

    # Plot a few time series
    df.iloc[:, :5].astype(float).plot(figsize=(12, 6))
    plt.tight_layout()
    plt.savefig(reports_dir / "sample_timeseries.png")
    plt.close()


# --- EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    # 1. Load raw data from artifacts config
    input_path = Path(config.artifacts.input_file)

    df = pd.read_csv(
        input_path,
        sep=';',
        header=None,
        engine='python',
        dtype=object,
        on_bad_lines='skip'
    )

    # 2. Convert first column to datetime and set as index
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.set_index(df.columns[0])

    # 3. Assign proper column names (MT_001 ... MT_370)
    df.columns = [f"MT_{i:03d}" for i in range(1, df.shape[1] + 1)]

    # 4. Convert all MT_* columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # 5. Run profiling + EDA
    profile_time_series(df)
    run_eda(df)
