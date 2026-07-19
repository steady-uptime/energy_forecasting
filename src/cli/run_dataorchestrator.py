# src/cli/run_dataorchestrator.py
from src.core.config_loader import Config

def test_real_pipeline():
    cfg = Config().load("config.yaml")

    orchestrator = DataOrchestrator(
        repo=LocalRepository(),
        ingestion=IngestionService(cfg.data),
        preprocessor=DataPreprocessor(cfg.data),
        engineer=FeatureEngineer(cfg.data),
        data_cfg=cfg.data,
        artifacts_cfg=cfg.artifacts,
        run_cfg=cfg.run,
    )

    df = orchestrator.run_pipeline(filename=cfg.data.input_file)
    print(df.head())
