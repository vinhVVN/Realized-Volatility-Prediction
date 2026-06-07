import time
from contextlib import contextmanager
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("timer")

@contextmanager
def timer(name: str, logger_obj: Optional[object] = None):
    """
    Context manager để đo thời gian thực thi của một khối lệnh.

    Args:
        name (str): Tên của tiến trình cần đo.
        logger_obj (logging.Logger, optional): Logger để ghi thời gian. Nếu None, dùng logger mặc định.

    Yields:
        None
    """
    t0 = time.time()
    log = logger_obj if logger_obj else logger
    log.info(f"[{name}] Bắt đầu...")
    yield
    elapsed = time.time() - t0
    log.info(f"[{name}] Hoàn thành trong {elapsed:.4f} giây.")
