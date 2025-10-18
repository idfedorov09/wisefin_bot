from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message

default_router = Router()

class Registration(StatesGroup):
    waiting_for_name = State()
    waiting_for_age = State()


# === 2. Команда /start ===
@default_router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await message.answer("Привет! Как тебя зовут?")
    await state.set_state(Registration.waiting_for_name)


# === 3. Получаем имя ===
@default_router.message(Registration.waiting_for_name)
async def process_name(message: Message, state: FSMContext):
    name = message.text.strip()

    if len(name) < 2:
        await message.answer("Имя слишком короткое, попробуй снова 🙂")
        return

    await state.update_data(name=name)
    await message.answer(f"Отлично, {name}! Сколько тебе лет?")
    await state.set_state(Registration.waiting_for_age)


# === 4. Получаем возраст ===
@default_router.message(Registration.waiting_for_age)
async def process_age(message: Message, state: FSMContext):
    try:
        age = int(message.text)
    except ValueError:
        await message.answer("Пожалуйста, введи возраст числом 🔢")
        return

    user_data = await state.get_data()
    name = user_data["name"]

    await message.answer(f"Тебя зовут {name}, и тебе {age} лет. Приятно познакомиться! 🎉")

    # Завершаем FSM
    await state.clear()


# === 5. Команда /cancel для сброса ===
@default_router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Диалог сброшен. Можем начать заново с /start 🔄")