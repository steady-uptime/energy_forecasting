# src/core/config/factories/monitoring_factory.py
from ..schemas import MonitoringConfig, ReportingConfig
def build_reporting_config(raw: dict) -> ReportingConfig:
    return ReportingConfig(**raw)

def build_monitoring_config(raw: dict) -> MonitoringConfig:
    return MonitoringConfig(
        primary_metric=raw["primary_metric"],
        drift_threshold=raw["drift_threshold"],
        thresholds=raw["thresholds"],
        reporting=build_reporting_config(raw["reporting"])
    )