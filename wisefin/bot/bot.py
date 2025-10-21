from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommandScopeAllPrivateChats
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from wisefin.aiogram.fsm.storage.memory import ChatEventIsolation
from wisefin.aiogram.fsm.storage.sqlalchemy import SQLAlchemyStorage
from wisefin.bot.routers import configure_routers
from wisefin.settings import settings
from wisefin.commons.context import AppContext


async def on_startup():
    scheduler = AsyncIOScheduler()
    AppContext.set(scheduler)
    scheduler.start()


async def on_shutdown():
    scheduler = AppContext.get(AsyncIOScheduler)
    if scheduler is not None:
        scheduler.shutdown()


async def start_polling():
    bot = Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    AppContext.set(bot)

    dp = Dispatcher(
        storage=await SQLAlchemyStorage.create(
            url=settings.DB_URL,
            schema_mode=None,
        ),
        events_isolation=ChatEventIsolation()
    )

    configure_routers(dp)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    await commands_config()
    await dp.start_polling(bot)


async def commands_config():
    bot = AppContext.get(Bot)
    commands = [
        # TODO
        # BotCommand(command="bug",  description="Сообщить о баге"),
    ]

    await bot.set_my_commands(
        commands=commands,
        scope=BotCommandScopeAllPrivateChats()
    )
