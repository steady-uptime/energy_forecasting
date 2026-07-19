# src/infra/data_repository.py
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List
from loguru import logger

from src.core.exceptions import IngestionError
from src.core.config.schemas import DataConfig   # FIXED: Typed Contract import


class DataRepository:
    def __init__(self, project_root: Path, data_config: DataConfig, run_id: str):
        self.project_root = project_root
        self.data_config = data_config
        self.run_id = run_id

    def resolve_path(self, relative_path: str) -> Path:
        full_path = (self.project_root / relative_path).resolve()
        # Security Gate: Path Traversal Prevention
        if not str(full_path).startswith(str(self.project_root)):
            raise PermissionError(f"Security Breach: Access denied to path outside project root: {full_path}")
        return full_path

    def _resolve_dtypes_from_schema(self, columns: List[str]) -> Dict[str, str]:
        """
        Translates the abstract 'raw_schema' (Regex) into a 
        concrete dictionary for Pandas (Exact Matches).
        """
        dtype_map: Dict[str, str] = {}
        # Invariant: Accessing SchemaField objects via dot-notation
        raw_schema = self.data_config.raw_schema

        for entry in raw_schema:
            # RECONCILED: entry is a SchemaField object, not a dict
            if entry.column:
                # Handle exact matches (e.g., "timestamp")
                col_name = entry.column
                if col_name in columns:
                    dtype_map[col_name] = entry.type
            
            elif entry.column_pattern:
                # Handle regex matches (e.g., "^MT_\\d{3}$")
                pattern = entry.column_pattern
                target_type = entry.type
                for col in columns:
                    if re.match(pattern, col):
                        dtype_map[col] = target_type
        
        return dtype_map

    def read_csv(self, relative_path: str, sep: str = None) -> pd.DataFrame:
        # Invariant: Fallback to config if sep is not provided via orchestrator
        separator = sep if sep is not None else self.data_config.csv_separator
        path = self.resolve_path(relative_path)

        logger.bind(
            module="DataRepository",
            run_id=self.run_id,
            path=str(path)
        ).info("Attempting to read CSV")

        if not path.exists():
            logger.bind(
                module="DataRepository",
                run_id=self.run_id,
                path=str(path)
            ).error("File not found")
            raise IngestionError(
                "CSV file does not exist",
                context={"path": str(path)}
            )

        try:
            # STEP 1: Metadata Probe (Header Peek)
            header_df = pd.read_csv(path, sep=separator, header=0, nrows=0)
            columns = header_df.columns.tolist()
            
            # STEP 2: Contract Translation
            dtype_map = self._resolve_dtypes_from_schema(columns)

            logger.bind(
                module="DataRepository",
                run_id=self.run_id,
                columns=columns,
                dtype_map=dtype_map
            ).debug("Resolved dtypes from schema")

            # STEP 3: Data Ingestion with Type Enforcement
            df = pd.read_csv(
                path, 
                sep=separator, 
                dtype=dtype_map,
                parse_dates=['timestamp'] if 'timestamp' in columns else None
            )

            logger.bind(
                module="DataRepository",
                run_id=self.run_id,
                rows=len(df),
                cols=len(df.columns)
            ).info("CSV ingestion successful")

            return df

        except Exception as e:
            logger.bind(
                module="DataRepository",
                run_id=self.run_id,
                error=str(e)
            ).error("Data ingestion failure")
    
            raise IngestionError(
                "Failed to ingest CSV",
                context={
                    "path": str(path),
                    "sep": separator,
                    "dtype_map": dtype_map if 'dtype_map' in locals() else None
                }
            ) from e
