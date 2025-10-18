import inspect
import sys
import logging
from pathlib import Path
from loguru import logger

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>{process.name}</magenta>/<blue>{thread.name}</blue> | "
    "<level>{message}</level>{exception}"
)

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging(app_env: str, log_dir: Path, level: str | int = "DEBUG"):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()

    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=CONSOLE_FORMAT,
        backtrace=True,
        diagnose=(app_env != "prod"),
    )

    logger.add(
        log_dir / "app.log",
        level="DEBUG",
        rotation="100 MB",
        retention=0,
        enqueue=True,
        serialize=True,
        backtrace=True,
        diagnose=False,
    )

    logging.basicConfig(
        handlers=[InterceptHandler()], 
        level=level,
        force=True
    )
    
    for name in logging.Logger.manager.loggerDict.keys():
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False
