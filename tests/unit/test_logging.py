
import json

from pulseflow.core.logging import get_logger, setup_logging


def test_logging_creates_log_file(tmp_path):
    log_file = tmp_path / "test.log"

    setup_logging(level="INFO", log_file=str(log_file))

    logger = get_logger("test")
    logger.info("pipeline_started", component="data_engineering")

    assert log_file.exists()

    log_content = log_file.read_text(encoding="utf-8")
    log_entry = json.loads(log_content.strip().splitlines()[-1])

    assert log_entry["event"] == "pipeline_started"
    assert log_entry["component"] == "data_engineering"
    assert log_entry["level"] == "info"
