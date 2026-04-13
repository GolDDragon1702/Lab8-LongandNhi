import sys
from pathlib import Path
from loguru import logger

class LoggerConfig:
    LogDir = "./logs/"
    BackTrace = False
    MaxBytes = 10 * 1024 * 1024  # 10MB ~ 10485760 bytes
    MaxBackupCount = 10
    # Đặt False để lúc test đọc file text cho dễ, đặt True nếu muốn log cho hệ thống phân tích đọc
    SerializeJSON = False 
    Diagnose = False 

def setup_logger(name: str = "rag_app"):
    """Thiết lập cấu hình cho loguru."""
    # Tạo thư mục chứa log nếu chưa có
    logdir = Path(LoggerConfig.LogDir)
    logdir.mkdir(parents=True, exist_ok=True)
    file_path = logdir / name

    # Xóa handler mặc định của loguru để tránh bị in đúp log
    logger.remove()

    # 1. Console Handler: In ra màn hình (Có màu sắc cho dễ nhìn)
    logger.add(
        sys.stdout,
        level="INFO",
        backtrace=LoggerConfig.BackTrace,
        diagnose=LoggerConfig.Diagnose,
        enqueue=True, # Thread-safe, rất tốt cho FastAPI
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 2. File Handler: Lưu vào file app.log
    logger.add(
        file_path.with_suffix(".log"),
        level="INFO",
        rotation=LoggerConfig.MaxBytes,
        retention=LoggerConfig.MaxBackupCount,
        backtrace=LoggerConfig.BackTrace,
        diagnose=LoggerConfig.Diagnose,
        enqueue=True,
        serialize=LoggerConfig.SerializeJSON,
        encoding="utf-8"
    )

def get_logger():
    """Trả về đối tượng logger để các file khác import dùng."""
    return logger