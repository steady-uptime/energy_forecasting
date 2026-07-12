# src/alerts/email_alert.py
# src/alerts/email_alert.py

from loguru import logger

DEFAULT_EMAIL_TO = "mlops-team@example.com"
DEFAULT_EMAIL_FROM = "mlops-system@example.com"

def send_email_alert(report: dict):
    """
    Placeholder email alert.
    Action is commented out — only logs the intent.
    """

    metric = report.get("metric_name", "unknown")
    baseline = report.get("baseline", "unknown")
    current = report.get("current", "unknown")

    logger.info(
        f"Email alert sent to {DEFAULT_EMAIL_TO} "
        f"(from: {DEFAULT_EMAIL_FROM}) — "
        f"metric={metric}, baseline={baseline}, current={current}"
    )

    # Example real email call (commented out)
    # import smtplib
    # from email.mime.text import MIMEText
    # msg = MIMEText(f"Drift detected: {metric} baseline={baseline}, current={current}")
    # msg["Subject"] = "Drift Detected"
    # msg["From"] = DEFAULT_EMAIL_FROM
    # msg["To"] = DEFAULT_EMAIL_TO
    # with smtplib.SMTP("localhost") as server:
    #     server.send_message(msg)
