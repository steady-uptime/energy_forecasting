# pipelines/monitoring_trigger.py
# pipelines/monitoring_trigger.py
import sys
import json
from pathlib import Path
from loguru import logger
import subprocess

from src.alerts.slack_alert import send_slack_alert
from src.alerts.email_alert import send_email_alert

MONITORING_DIR = Path("artifacts/monitoring")

def load_latest_report():
    reports = sorted(MONITORING_DIR.glob("monitoring_*.json"))
    if not reports:
        raise RuntimeError("No monitoring reports found.")
    latest = reports[-1]
    logger.info(f"Loaded monitoring report: {latest}")
    return json.loads(latest.read_text())

def trigger_retrain():
    logger.info("Triggering retrain pipeline...")
    import sys
    subprocess.run([sys.executable, "-m", "pipelines.retrain_pipeline"], check=True)
    logger.info("Retrain pipeline completed.")

def main():
    report = load_latest_report()
    drift = report.get("drift_detected", False)

    if drift:
        logger.warning("Drift detected — sending alerts and triggering retraining")

        # Optional alerts (logged only)
        send_slack_alert(report)
        send_email_alert(report)

        # Retrain
        trigger_retrain()

    else:
        logger.info("No drift detected — no action taken")

if __name__ == "__main__":
    main()
