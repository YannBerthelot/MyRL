from app.logger import create_logger
import os

os.makedirs("logs", exist_ok=True)

logger = create_logger(
    log_folder_path="logs/logs",
    log_stream_level="ERROR",
    log_write_level="DEBUG",
)