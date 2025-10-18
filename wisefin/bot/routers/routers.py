from aiogram import Dispatcher

from wisefin.bot.routers.impl.default_router import default_router

def configure_routers(dp: Dispatcher):
    # dp.update.outer_middleware(TransactionMiddleware())
    dp.include_routers(default_router)
