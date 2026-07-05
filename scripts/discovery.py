# scripts/discovery.py
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from src.core.config_loader import config
from src.infra.logger import setup_logger


def load_raw_data() -> pd.DataFrame:
    setup_logger(config.logging, Path(config.project_root))
    logger.info("System: Production logger initialized. Starting training pipeline.")
    
    """
    Loads the raw dataset using paths resolved via the Configuration Singleton.
    Enforces data types immediately to ensure system integrity.
    """
    raw_path = config.paths.raw_data / "LD2011_2014.txt"
    
    logger.info(f"System: Attempting to load data from {raw_path}")
    
    if not raw_path.exists():
        logger.error(f"System: Data file not found at {raw_path}")
        raise FileNotFoundError(f"Path {raw_path} does not exist.")

    try:
        # Dataset Specifics: Semi-colon delimiter, header=None
        df = pd.read_csv(raw_path, sep=";", header=None, low_memory=False)
        logger.info(f"System: Successfully loaded {len(df)} rows.")

        # --- TYPE ENFORCEMENT GATE ---
        # Column 0 is 'timestamp'. We force it to datetime64.
        # We also cast all other columns to numeric to ensure math operations work.
        df[0] = pd.to_datetime(df[0], errors='coerce')
        
        # Dynamically cast all other columns to numeric (float64)
        # This handles the consumption data for all 370 clients at once.
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename for clarity (Standardizing the schema for discovery)
        df.columns = ['timestamp'] + [f'client_{i}' for i in range(1, df.shape[1])]
        
        return df
    except Exception as e:
        logger.error(f"System: Failed to parse CSV: {e}")
        raise

# def profile_time_series(df: pd.DataFrame) -> None:
#     """
#     Analyzes the dataset to identify DST anomalies and interval consistency.
#     This is the 'Discovery' logic.
#     """
#     logger.info("System: Starting time-series profiling...")
#     
#     # Rename columns for readability during discovery
#     # Column 0 is 'timestamp', others are 'client_id'
#     df.columns = ['timestamp'] + [f'client_{i}' for i in range(1, df.shape[1])]
#     
#     # Convert timestamp to datetime objects
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     
#     # 1. Check for 15-minute interval consistency
#     # We calculate the difference between consecutive rows for the first client
#     sample_client = df.iloc[:, 1]
#     intervals = df['timestamp'].diff().dt.total_seconds() / 900  # 15 minutes = 900 seconds
#     
#     anomalies = intervals[intervals != 1.0]
#     if not anomalies.empty:
#         logger.warning(f"System: Found {len(anomalies)} instances of non-15-minute intervals.")
#     else:
#         logger.info("System: All intervals are consistent (15-minute).")
# 
#     # 2. Identify DST anomalies (The March/October rules)
#     # We look for jumps in the sequence
#     # Log the first and last few rows to identify the exact format of the jump
#     logger.info("System: Sampling jump points for DST analysis...")
#     # This is where you manually observe the 23h vs 25h jumps to verify the rules
#     print(df.head(10)) # Direct print is acceptable in discovery scripts

def profile_time_series(df: pd.DataFrame) -> None:
    """
    Performs a diagnostic dump of the dataset to characterize the layout,
    types, and dimensions.
    """
    logger.info("System: Starting Diagnostic Profiling...")
    
    # 1. Print Dimensions
    # This helps us verify the "370 instances / 140,256 features" claim
    print(f"\n--- DIMENSIONS ---")
    print(f"Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")

    # 2. Print Data Types and Null Counts
    # This is where we find out why the division is failing.
    print(f"\n--- DATA TYPES & NULLS ---")
    print(df.info())

    # 3. Print Sample Data
    # We'll look at the first 5 rows. 
    # Note: If there are 140k columns, this will be truncated by the terminal.
    print(f"\n--- DATA SAMPLE (First 5 Rows) ---")
    print(df.head())

    # 4. Check for Interval Consistency (with added error handling)
    # We'll try to do the calculation but catch the error to see the state.
    try:
        # Ensure timestamp is converted to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Calculate differences
        diffs = df['timestamp'].diff()
        
        # Check if the result is actually numeric
        if pd.api.types.is_numeric_dtype(diffs.dt.total_seconds):
            intervals = diffs.dt.total_seconds() / 900
            anomalies = intervals[intervals != 1.0].count()
            logger.info(f"System: Interval consistency check passed. Anomalies found: {anomalies}")
        else:
            logger.error("System: .dt.total_seconds() produced non-numeric output.")
    except Exception as e:
        logger.error(f"System: Interval check failed: {e}")

def main() -> None:
    try:
        df = load_raw_data()
        profile_time_series(df)
    except Exception as e:
        logger.critical(f"System: Discovery Pipeline Failed: {e}")

if __name__ == "__main__":
    main()
