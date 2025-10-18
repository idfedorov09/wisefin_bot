import threading
from contextvars import ContextVar
from contextlib import asynccontextmanager
from typing import Dict, Type, Any, TypeVar, Optional, cast

T = TypeVar("T")


class AsyncContext:
    """
    Асинхронный контекст выполнения (привязан к задаче/запросу через ContextVar).
    """

    # Храним текущий контекст (словарь "класс -> объект")
    _request_context: ContextVar[Dict[Type[Any], Any]] = ContextVar(
        "request_context",
        default={}
    )

    @classmethod
    async def set(cls, obj: T) -> None:
        """
        Асинхронно сохраняет объект в контекст по ключу = тип объекта.

        Пример:
            user = User(...)
            await AsyncContext.set(user)
        """
        old_ctx = cls._request_context.get()
        new_ctx = dict(old_ctx)
        new_ctx[obj.__class__] = obj
        cls._request_context.set(new_ctx)

    @classmethod
    async def get(cls, obj_type: Type[T]) -> Optional[T]:
        """
        Асинхронно возвращает объект из контекста по классу (если он есть).

        Пример:
            user = await AsyncContext.get(User)
        """
        return cast(Optional[T], cls._request_context.get().get(obj_type))

    @classmethod
    async def has(cls, obj_type: Type[Any]) -> bool:
        """
        Проверяет, хранится ли в контексте объект указанного класса.

        Пример:
            if await AsyncContext.has(User):
                print("Объект User есть в контексте")
        """
        return obj_type in cls._request_context.get()

    @classmethod
    async def clear(cls) -> None:
        """
        Полностью очищает контекст.
        """
        cls._request_context.set({})

    @classmethod
    async def get_all(cls) -> Dict[Type[Any], Any]:
        """
        Возвращает копию всего текущего словаря контекста (для отладки).
        """
        return dict(cls._request_context.get())

    @classmethod
    @asynccontextmanager
    async def temp_set(cls, obj: T):
        """
        Асинхронный контекст-менеджер для «временного» сохранения объекта.
        После выхода из `async with` контекст откатится обратно.

        Пример использования:
            async with AsyncContext.temp_set(MyObject(...)):
                # внутри блока в контексте есть MyObject(...)
                ...
            # здесь контекст вернулся в исходное состояние
        """
        old_ctx = cls._request_context.get()
        new_ctx = dict(old_ctx)
        new_ctx[obj.__class__] = obj
        cls._request_context.set(new_ctx)

        try:
            yield
        finally:
            cls._request_context.set(old_ctx)

    @classmethod
    @asynccontextmanager
    async def isolate_context(cls, new_value: Optional[Dict[Type[Any], Any]] = None):
        """
        Асинхронный контекст-менеджер для изоляции контекста.
        Если new_value не передан, берётся копия текущего контекста.
        Все изменения внутри блока не затрагивают внешний контекст.

        Пример использования:
            async with AsyncContext.isolate_context():
                # внутри блока можно менять контекст, а снаружи он останется прежним
                ...
        """
        old_ctx = cls._request_context.get()
        if new_value is None:
            new_value = dict(old_ctx)
        cls._request_context.set(new_value)
        try:
            yield
        finally:
            cls._request_context.set(old_ctx)

    @classmethod
    async def delete(cls, obj_type: T) -> None:
        """
        Асинхронно удаляет объект из контекста по его классу, если он присутствует.

        Пример:
            await AsyncContext.remove(User)
        """
        old_ctx = cls._request_context.get()
        new_ctx = dict(old_ctx)
        new_ctx.pop(obj_type, None)
        cls._request_context.set(new_ctx)


class AppContext:
    """
    Глобальный контекст приложения (потокобезопасный).
    """
    _global_context: Dict[Type[Any], Any] = {}
    _lock = threading.RLock()

    @classmethod
    def set(cls, obj: T) -> None:
        """
        Сохраняет объект в контексте по ключу = тип объекта.

        Пример:
            user = User(...)
            AppContext.set(user)
        """
        with cls._lock:
            cls._global_context[obj.__class__] = obj

    @classmethod
    def get(cls, obj_type: Type[T]) -> Optional[T]:
        """
        Возвращает объект из контекста по классу (если он есть).

        Пример:
            user = AppContext.get(User)
        """
        with cls._lock:
            return cls._global_context.get(obj_type)

    @classmethod
    def has(cls, obj_type: Type[Any]) -> bool:
        """
        Проверяет, хранится ли в контексте объект указанного класса.

        Пример:
            if AppContext.has(User):
                print("Объект User есть в контексте")
        """
        with cls._lock:
            return obj_type in cls._global_context

    @classmethod
    def clear(cls) -> None:
        """
        Очищает весь контекст.
        """
        with cls._lock:
            cls._global_context.clear()

    @classmethod
    def get_all(cls) -> Dict[Type[Any], Any]:
        """
        Возвращает копию всего текущего словаря контекста (для отладки).
        """
        with cls._lock:
            return dict(cls._global_context)