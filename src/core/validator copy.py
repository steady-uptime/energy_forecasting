# src/core/validator.py
# Validate the schema in the orchestrator, not the service modifying the schema.
# This is pure Separation of Concerns + Service Contract Enforcement.
# Run DataValidator after each:
# IngestionService (raw_schema)
# DataPreprocessor (preprocessed_schema - drop_columns)
# FeatureEngineer (engineered_schema - transform, drop non-numeric columns)
# Do not validate after DataSplitter

from loguru import logger
import pandas as pd
import re
from typing import Dict, List, Union

class DataValidator:
    # Define common type aliases to make the validator robust
    # This handles differences between Pandas versions and SQL types
    TYPE_MAPPINGS = {
        "object": ["object", "str"],
        "str": ["object", "str"],
        "float": ["float64", "float32", "float"],
        "int": ["int64", "int32", "int"]
    }
    
    def __init__(self, expected_schema: Union[Dict[str, str], List[Dict[str, str]]]):
        """
        Supports:
        - {"column": "name", "type": "dtype"}
        - {"column_pattern": "regex", "type": "dtype"}
        """
        self.explicit_columns = {}
        self.pattern_columns = []

        if isinstance(expected_schema, list):
            for item in expected_schema:
                if "column" in item:
                    self.explicit_columns[item["column"]] = item["type"]
                elif "column_pattern" in item:
                    self.pattern_columns.append(
                        {"pattern": item["column_pattern"], "type": item["type"]}
                    )
                else:
                    raise KeyError("Schema entry must contain 'column' or 'column_pattern'.")
        else:
            self.explicit_columns = expected_schema

        logger.debug(
            f"Validator initialized.\n"
            f"Explicit: {self.explicit_columns}"
            f"Patterns: {self.pattern_columns}"
        )

    def validate(self, df: pd.DataFrame) -> None:
        logger.info("Starting contract validation...")

        # Expand pattern-based columns
        expanded = {}
        for entry in self.pattern_columns:
            regex = re.compile(entry["pattern"])
            matches = [col for col in df.columns if regex.match(col)]
            for col in matches:
                expanded[col] = entry["type"]

        # Merge explicit + expanded
        full_schema = {**self.explicit_columns, **expanded}

        # Check missing columns
        missing = set(full_schema.keys()) - set(df.columns)
        if missing:
            logger.error(f"Missing columns: {missing}")
            raise RuntimeError(f"Contract Validation Failed: Missing columns: {missing}")

        # Validate dtypes
        for col, expected_type in full_schema.items():
            actual_type = str(df[col].dtype)

            valid = False
            if expected_type in self.TYPE_MAPPINGS and actual_type in self.TYPE_MAPPINGS[expected_type]:
                valid = True
            elif expected_type in actual_type:
                valid = True

            if not valid:
                raise RuntimeError(
                    f"Contract Validation Failed: Column '{col}' expected {expected_type}, got {actual_type}"
                )

        logger.success("Contract validation passed.")