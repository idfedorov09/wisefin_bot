from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from enum import Enum
from typing import Any, Dict, Mapping, Optional, cast, Literal, Union
from aiogram.exceptions import DataNotDictLikeError
from aiogram.fsm.state import State
from aiogram.fsm.storage.base import (
    BaseStorage,
    DefaultKeyBuilder,
    KeyBuilder,
    StateType,
    StorageKey,
)
from sqlalchemy import String, DateTime, func, inspect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON as SAJSON

try:
    from sqlalchemy.dialects.postgresql import JSONB

    JSON_AUTO = SAJSON().with_variant(JSONB, "postgresql")
except Exception:
    JSON_AUTO = SAJSON()


class SchemaMode(str, Enum):
    NONE = "none"
    VALIDATE = "validate"
    UPDATE = "update"
    CREATE = "create"
    CREATE_DROP = "create-drop"

_Event = Literal["start", "stop"]

# TODO: set to VALIDATE
_default_schema_mode = SchemaMode(
    os.getenv("AIOGRAM_SQLALCHEMY_STORAGE_SCHEMA_MODE", SchemaMode.CREATE_DROP.value)
)

class Base(DeclarativeBase):
    pass


class FSMRow(Base):
    __tablename__ = "aiogram_fsm"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    state: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON_AUTO, nullable=True)
    updated_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class BaseSessionContext(ABC):
    """Базовый интерфейс для фабрик асинхронных контекстных менеджеров."""

    @abstractmethod
    def __call__(self) -> "BaseSessionContext":
        """Возвращает новый контекстный менеджер (обычно self или новый экземпляр)."""
        raise NotImplementedError

    @abstractmethod
    async def __aenter__(self) -> AsyncSession:
        """Вход в контекст, возвращает асинхронную сессию."""
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Выход из контекста, выполняет commit/rollback."""
        raise NotImplementedError


class SimpleSessionContext(BaseSessionContext):
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], use_transaction: bool = True):
        self._sessionmaker = sessionmaker
        self._use_transaction = use_transaction
        self._session: Optional[AsyncSession] = None
        self._transaction = None

    def __call__(self) -> "SimpleSessionContext":
        return SimpleSessionContext(self._sessionmaker, self._use_transaction)

    async def __aenter__(self) -> AsyncSession:
        self._session = self._sessionmaker()
        try:
            if self._use_transaction:
                self._transaction = await self._session.begin()
            return self._session
        except:
            with suppress(Exception):
                await self._session.close()
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._use_transaction and self._transaction is not None:
                if exc_type:
                    with suppress(Exception):
                        await self._transaction.rollback()
                else:
                    await self._transaction.commit()
        finally:
            if self._session is not None:
                await self._session.close()


class SQLAlchemyStorage(BaseStorage):
    """
    FSM storage для aiogram 3.x на SQLAlchemy 2.x (async ORM).
    Поведение:
      - set_state(None) / set_data({}) очищают поле
      - если state is None и (not data), строка удаляется
    """

    def __init__(
            self,
            engine: AsyncEngine,
            *,
            key_builder: Optional[KeyBuilder] = None,
            schema_mode: Optional[Union[SchemaMode, str]] = _default_schema_mode,
            table_name: str = "aiogram_fsm",
            session_context: Optional[BaseSessionContext] = None,
            sessionmaker_kwargs: Optional[Dict[str, Any]] = None,
            use_transaction: bool = True,
    ) -> None:
        if table_name != FSMRow.__tablename__:
            FSMRow.__tablename__ = table_name  # type: ignore[attr-defined]
        if isinstance(schema_mode, str):
            schema_mode = SchemaMode(schema_mode)

        self._key_builder: KeyBuilder = key_builder or DefaultKeyBuilder()
        self._engine: AsyncEngine = engine
        self._session_context = session_context \
                                or SQLAlchemyStorage._make_simple_session_context(
            engine,
            sessionmaker_kwargs,
            use_transaction
        )
        self._schema_mode = schema_mode or SchemaMode.NONE
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _make_simple_session_context(
            engine: AsyncEngine,
            sessionmaker_kwargs: Optional[Dict[str, Any]] = None,
            use_transaction: bool = True,
    ) -> SimpleSessionContext:
        sm_kwargs = {"expire_on_commit": False}
        if sessionmaker_kwargs:
            sm_kwargs.update(sessionmaker_kwargs)
        return SimpleSessionContext(
            sessionmaker=async_sessionmaker(engine, **sm_kwargs),
            use_transaction=use_transaction
        )

    @classmethod
    async def from_url(
            cls,
            url: str,
            *,
            key_builder: Optional[KeyBuilder] = None,
            session_context: Optional[BaseSessionContext] = None,
            table_name: str = "aiogram_fsm",
            schema_mode: Optional[Union[SchemaMode, str]] = _default_schema_mode,
            engine_kwargs: Optional[Dict[str, Any]] = None,
            sessionmaker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SQLAlchemyStorage":
        engine = create_async_engine(url, **(engine_kwargs or {}))
        storage = cls(
            engine=engine,
            key_builder=key_builder,
            session_context=session_context,
            table_name=table_name,
            sessionmaker_kwargs=sessionmaker_kwargs,
            schema_mode=schema_mode
        )
        await storage.schema_event("start")
        return storage

    async def schema_event(self, event: _Event) -> None:
        # TODO: более красивая реализация?
        # TODO: работать ТОЛЬКО с таблицей fsm
        if self._schema_mode == SchemaMode.NONE:
            return
        elif self._schema_mode == SchemaMode.CREATE:
            async with self._engine.begin() as conn:
                if event == "start":
                    self._logger.debug("Recreate fsm schema (drop and create)")
                    await conn.run_sync(Base.metadata.drop_all)
                    await conn.run_sync(Base.metadata.create_all)
        elif self._schema_mode == SchemaMode.CREATE_DROP:
            async with self._engine.begin() as conn:
                if event == "start":
                    self._logger.debug("Create fsm schema")
                    await conn.run_sync(Base.metadata.create_all)
                elif event == "stop":
                    self._logger.debug("Drop fsm schema")
                    await conn.run_sync(Base.metadata.drop_all)
        elif self._schema_mode == SchemaMode.UPDATE:
            raise NotImplementedError("Implement using alembic")
        elif self._schema_mode == SchemaMode.VALIDATE:
            raise NotImplementedError("Check the fact fsm_table is created with correct schema")


    @staticmethod
    def resolve_state(value: StateType) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, State):
            return value.state
        return str(value)

    @staticmethod
    async def _delete_if_empty(session: AsyncSession, row: Optional[FSMRow]) -> None:
        """
        Удаляет запись FSMRow из базы, если она пуста.

        Запись считается пустой, если:
        - поле `state` равно None
        - поле `data` отсутствует или пустое

        :param session: Активная асинхронная сессия SQLAlchemy.
        :param row: Экземпляр FSMRow (или None), который необходимо проверить и при необходимости удалить.
        """
        if row is None:
            return
        if (row.state is None) and (not row.data):
            await session.delete(row)

    async def set_state(self, key: StorageKey, state: StateType = None) -> None:
        row_id = self._key_builder.build(key)
        resolved = self.resolve_state(state)

        async with self._session_context() as session:
            row: Optional[FSMRow] = await session.get(FSMRow, row_id)

            if resolved is None:
                if row is None:
                    return
                row.state = None
                await self._delete_if_empty(session, row)
                return

            if row is None:
                session.add(FSMRow(id=row_id, state=resolved, data=None))
            else:
                row.state = resolved

    async def get_state(self, key: StorageKey) -> Optional[str]:
        row_id = self._key_builder.build(key)
        async with self._session_context() as session:
            row = await session.get(FSMRow, row_id)
            return cast(Optional[str], row.state if row else None)

    async def set_data(self, key: StorageKey, data: Mapping[str, Any]) -> None:
        if not isinstance(data, dict):
            raise DataNotDictLikeError(
                f"Data must be a dict or dict-like object, got {type(data).__name__}"
            )

        row_id = self._key_builder.build(key)
        payload: Optional[Dict[str, Any]] = dict(data) if data else None

        async with self._session_context() as session:
            row: Optional[FSMRow] = await session.get(FSMRow, row_id)

            if payload is None:
                if row is None:
                    return
                row.data = None
                await self._delete_if_empty(session, row)
                return

            if row is None:
                session.add(FSMRow(id=row_id, state=None, data=payload))
            else:
                row.data = payload

    async def get_data(self, key: StorageKey) -> Dict[str, Any]:
        row_id = self._key_builder.build(key)
        async with self._session_context() as session:
            row = await session.get(FSMRow, row_id)
            return cast(Dict[str, Any], row.data or {}) if row else {}

    async def close(self) -> None:  # pragma: no cover
        await self.schema_event("stop")
        await self._engine.dispose()
