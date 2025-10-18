import sys
from pathlib import Path
from loguru import logger

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>{process.name}</magenta>/<blue>{thread.name}</blue> | "
    "<level>{message}</level>\n{exception}"
)

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
