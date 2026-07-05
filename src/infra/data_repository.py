# src/infra/data_repository.py
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from src.core.exceptions import IngestionError
from loguru import logger

class DataRepository:
    def __init__(self, project_root: Path, data_config: Any, run_id: str):
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
        raw_schema = self.data_config.raw_schema

        for entry in raw_schema:
            if "column" in entry:
                # Handle exact matches (e.g., "timestamp")
                col_name = entry["column"]
                if col_name in columns:
                    dtype_map[col_name] = entry["type"]
            
            elif "column_pattern" in entry:
                # Handle regex matches (e.g., "^MT_\\d{3}$")
                pattern = entry["column_pattern"]
                target_type = entry["type"]
                for col in columns:
                    if re.match(pattern, col):
                        dtype_map[col] = target_type
        
        return dtype_map

    def read_csv(self, relative_path: str, sep: str = ",") -> pd.DataFrame:
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
            # We read 0 rows to extract column names without loading the data.
            header_df = pd.read_csv(path, sep=sep, header=0, nrows=0)
            columns = header_df.columns.tolist()
            
            # STEP 2: Contract Translation
            # Convert the YAML Regex patterns into a flat Dict for Pandas.
            dtype_map = self._resolve_dtypes_from_schema(columns)

            logger.bind(
                module="DataRepository",
                run_id=self.run_id,
                columns=columns,
                dtype_map=dtype_map
            ).debug("Resolved dtypes from schema")

            # STEP 3: Data Ingestion with Type Enforcement
            # We pass the dtypes to ensure the 'Execution' matches our 'Intent'.
            # We also handle date parsing for the 'timestamp' column specifically.
            df = pd.read_csv(
                path, 
                sep=sep, 
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
                    "sep": sep,
                    "dtype_map": dtype_map if 'dtype_map' in locals() else None
                }
            ) from e
