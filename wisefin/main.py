import asyncio
from loguru import logger

from wisefin.settings import settings, setup_logging


async def main():
    setup_logging(
        app_env=settings.APP_ENV,
        log_dir=settings.LOG_DIR_PATH,
        level=settings.LOG_LEVEL
    )
    logger.info("Started")


if __name__ == "__main__":
    asyncio.run(main())
