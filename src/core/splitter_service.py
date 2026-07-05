# src/core/splitter_service.py
import pandas as pd
from src.core.exceptions import SplitError, DataValidationError
from loguru import logger

class TimeSeriesSplitter:
    def __init__(self, split_cfg: dict, target_column: str, run_id: str):
        self.split_cfg = split_cfg
        self.target_column = target_column
        self.run_id = run_id

        logger.bind(
            module="TimeSeriesSplitter",
            run_id=self.run_id,
            target_column=self.target_column
        ).info("TimeSeriesSplitter initialized")

    def split(self, df: pd.DataFrame):
        logger.bind(
            module="TimeSeriesSplitter",
            run_id=self.run_id,
            rows=len(df),
            cols=len(df.columns)
        ).info("Starting time-series split")

        try:
            if df.empty:
                raise DataValidationError(
                    "Cannot split an empty DataFrame.",
                    context={"rows": 0, "columns": []}
                )

            if "timestamp" not in df.columns:
                raise SplitError(
                    "Timestamp column missing for time-series split.",
                    context={"columns": list(df.columns)}
                )

            train_end = pd.to_datetime(self.split_cfg["train_end"])
            val_end = pd.to_datetime(self.split_cfg["val_end"])
            test_end = pd.to_datetime(self.split_cfg["test_end"])

            logger.bind(
                module="TimeSeriesSplitter",
                run_id=self.run_id,
                train_end=str(train_end),
                val_end=str(val_end),
                test_end=str(test_end)
            ).debug("Split boundaries resolved")

            # 1. Perform the Temporal Split
            train = df[df["timestamp"] <= train_end]
            val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
            test = df[df["timestamp"] > val_end]

            # 2. Define the Feature Set
            drop_cols = [self.target_column, "timestamp"]
            cols_to_drop = [c for c in drop_cols if c in train.columns]

            X_train = train.drop(columns=cols_to_drop)
            y_train = train[self.target_column]

            X_test = test.drop(columns=cols_to_drop)
            y_test = test[self.target_column]

            # 3. Dynamic Schema Generation
            dynamic_schema = {
                col: str(dtype)
                for col, dtype in X_train.dtypes.items()
            }

            logger.bind(
                module="TimeSeriesSplitter",
                run_id=self.run_id,
                X_train_shape=X_train.shape,
                X_test_shape=X_test.shape,
                features=list(X_train.columns)
            ).info("Split complete")

            return X_train, y_train, X_test, y_test, dynamic_schema

        except Exception as e:
            logger.bind(
                module="TimeSeriesSplitter",
                run_id=self.run_id,
                error=str(e)
            ).error("Time-series split failure")

            raise SplitError(
                "Failed during time-series split",
                context={
                    "split_cfg": self.split_cfg,
                    "target_column": self.target_column
                }
            ) from e
