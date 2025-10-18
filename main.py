import asyncio

from wisefin.settings import settings, setup_logging
from wisefin.bot import start_polling

async def main():
    setup_logging(
        app_env=settings.APP_ENV,
        log_dir=settings.LOG_DIR_PATH,
        level=settings.LOG_LEVEL
    )
    await start_polling()


if __name__ == "__main__":
    asyncio.run(main())
