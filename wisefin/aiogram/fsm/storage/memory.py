import asyncio
from contextlib import asynccontextmanager
from typing import Dict

from aiogram.fsm.storage.base import BaseEventIsolation, StorageKey


class ChatEventIsolation(BaseEventIsolation):
    def __init__(self):
        self._locks: Dict[int, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock_for_chat(self, chat_id: int) -> asyncio.Lock:
        async with self._global_lock:
            lock = self._locks.get(chat_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[chat_id] = lock
            return lock

    @asynccontextmanager
    async def lock(self, key: StorageKey):
        """
        Блокирует все обработчики с тем же chat_id до выхода из контекста.
        """
        lock = await self._get_lock_for_chat(key.chat_id)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
            async with self._global_lock:
                if not lock.locked():
                    self._locks.pop(key.chat_id, None)

    async def close(self) -> None:
        """
        Очистка — снимаем все locks.
        """
        async with self._global_lock:
            self._locks.clear()