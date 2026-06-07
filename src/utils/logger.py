import logging
import sys
from pathlib import Path

def get_logger(name: str = "optiver", log_file: str = None) -> logging.Logger:
    """
    Tạo và cấu hình logger với định dạng chuẩn enterprise.

    Args:
        name (str): Tên của logger.
        log_file (str, optional): Đường dẫn tới file để ghi log. Nếu None, chỉ in ra console.

    Returns:
        logging.Logger: Đối tượng logger đã cấu hình.
    """
    logger = logging.getLogger(name)
    
    # Nếu logger đã có handler, tránh ghi đúp
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
