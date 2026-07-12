# src/alerts/slack_alert.py
# src/alerts/slack_alert.py

from loguru import logger

DEFAULT_SLACK_CHANNEL = "#mlops-alerts"
DEFAULT_SLACK_USER = "mlops-system"

def send_slack_alert(report: dict):
    """
    Placeholder Slack alert.
    Action is commented out — only logs the intent.
    """

    metric = report.get("metric_name", "unknown")
    baseline = report.get("baseline", "unknown")
    current = report.get("current", "unknown")

    logger.info(
        f"Slack alert sent to {DEFAULT_SLACK_CHANNEL} "
        f"(user: {DEFAULT_SLACK_USER}) — "
        f"metric={metric}, baseline={baseline}, current={current}"
    )

    # Example real Slack call (commented out)
    # import requests
    # webhook = "<your webhook>"
    # message = {"text": f"Drift detected: {metric} baseline={baseline}, current={current}"}
    # requests.post(webhook, json=message)
