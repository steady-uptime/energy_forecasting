# src/core/feature_engineering.py
import pandas as pd
from src.core.config_loader import FeaturesConfig
from src.core.exceptions import FeatureEngineeringError
from loguru import logger

class FeatureEngineer:
    def __init__(self, rules: FeaturesConfig, target_column: str, run_id: str):
        self.rules = rules
        self.target_column = target_column
        self.run_id = run_id

        logger.bind(
            module="FeatureEngineer",
            run_id=self.run_id,
            target_column=self.target_column
        ).info("FeatureEngineer initialized")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.bind(
            module="FeatureEngineer",
            run_id=self.run_id,
            rows=len(df),
            cols=len(df.columns)
        ).info("Starting feature engineering")

        try:
            df = df.copy()

            # -------------------------------------------------
            # Total Load Feature
            # -------------------------------------------------
            total_cfg = self.rules.total_load
            pattern = total_cfg["pattern"]
            method = total_cfg["method"]

            mt_cols = df.filter(regex=pattern).columns
            df["total_load"] = df[mt_cols].sum(axis=1).astype("float64")

            logger.bind(
                module="FeatureEngineer",
                run_id=self.run_id,
                mt_columns=list(mt_cols)
            ).debug("Created total_load feature")

            # -------------------------------------------------
            # Time-Based Features
            # -------------------------------------------------
            time_cfg = self.rules.time_features

            if time_cfg.get("hour_of_day", False):
                df["hour_of_day"] = df["timestamp"].dt.hour.astype("int64")

            if time_cfg.get("day_of_week", False):
                df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int64")

            if time_cfg.get("is_weekend", False):
                df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype("int64")

            logger.bind(
                module="FeatureEngineer",
                run_id=self.run_id,
                time_features=time_cfg
            ).debug("Created time-based features")

            # -------------------------------------------------
            # Final Contract Enforcement
            # -------------------------------------------------
            keep_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if "timestamp" in df.columns:
                keep_cols.append("timestamp")

            if self.target_column in df.columns and self.target_column not in keep_cols:
                keep_cols.append(self.target_column)

            final_df = df[keep_cols]

            dropped_cols = df.columns.difference(keep_cols)
            if len(dropped_cols) > 0:
                logger.bind(
                    module="FeatureEngineer",
                    run_id=self.run_id,
                    dropped_columns=list(dropped_cols)
                ).debug("Dropped non-numeric columns")

            logger.bind(
                module="FeatureEngineer",
                run_id=self.run_id,
                rows=len(final_df),
                cols=len(final_df.columns)
            ).info("Feature engineering completed")

            return final_df

        except Exception as e:
            logger.bind(
                module="FeatureEngineer",
                run_id=self.run_id,
                error=str(e)
            ).error("Feature engineering failure")

            raise FeatureEngineeringError(
                "Failed during feature engineering",
                context={
                    "rules": self.rules.dict() if hasattr(self.rules, "dict") else str(self.rules),
                    "target_column": self.target_column
                }
            ) from e
