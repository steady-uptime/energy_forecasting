





"""
The Experiment Tracker (The "MLOps" Way)
This is the most "Senior" move. When your train_pipeline.py runs, it should log the data metadata
  to your Experiment Tracker (e.g., MLflow, W&B, or your own experiment_tracker.py).

In your src/infra/experiment_tracker.py (or wherever you handle logging), ensure that every "Run" logs:

data_version: (e.g., a hash of the raw file or a timestamp)
data_license: "CC BY 4.0"
data_source: "UCI Repository"

The logic: If someone looks at your logs six months from now and sees a model with a specific RMSE, 
  they can see exactly which dataset, version, and license were used to produce that specific result.
"""