from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, cast

from aiogram.exceptions import DataNotDictLikeError
from aiogram.fsm.state import State
from aiogram.fsm.storage.base import (
    BaseStorage,
    DefaultKeyBuilder,
    KeyBuilder,
    StateType,
    StorageKey,
)

from sqlalchemy import String, DateTime, func
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
            table_name: str = "aiogram_fsm",
            create_table: bool = False,
            sessionmaker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Позволяем смену имени таблицы (если надо)
        if table_name != FSMRow.__tablename__:
            FSMRow.__tablename__ = table_name  # type: ignore[attr-defined]

        self._engine: AsyncEngine = engine
        self._key_builder: KeyBuilder = key_builder or DefaultKeyBuilder()

        sm_kwargs = {"expire_on_commit": False}
        if sessionmaker_kwargs:
            sm_kwargs.update(sessionmaker_kwargs)

        self._sessionmaker = async_sessionmaker(self._engine, **sm_kwargs)
        self._create_table_on_init = create_table

    # ---------- Фабрика ----------

    @classmethod
    async def from_url(
            cls,
            url: str,
            *,
            key_builder: Optional[KeyBuilder] = None,
            table_name: str = "aiogram_fsm",
            create_table: bool = True,
            engine_kwargs: Optional[Dict[str, Any]] = None,
            sessionmaker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SQLAlchemyStorage":
        engine = create_async_engine(url, **(engine_kwargs or {}))
        storage = cls(
            engine=engine,
            key_builder=key_builder,
            table_name=table_name,
            create_table=create_table,
            sessionmaker_kwargs=sessionmaker_kwargs,
        )
        if create_table:
            await storage.create_tables()
        return storage

    async def create_tables(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # ---------- Вспомогательные ----------

    @staticmethod
    def resolve_state(value: StateType) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, State):
            return value.state
        return str(value)

    async def _delete_if_empty(self, session: AsyncSession, row: Optional[FSMRow]) -> None:
        if row is None:
            return
        if (row.state is None) and (not row.data):
            await session.delete(row)

    # ---------- Реализация BaseStorage ----------

    async def set_state(self, key: StorageKey, state: StateType = None) -> None:
        row_id = self._key_builder.build(key)
        resolved = self.resolve_state(state)

        async with self._sessionmaker() as session, session.begin():
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
        async with self._sessionmaker() as session:
            row = await session.get(FSMRow, row_id)
            return cast(Optional[str], row.state if row else None)

    async def set_data(self, key: StorageKey, data: Mapping[str, Any]) -> None:
        if not isinstance(data, dict):
            raise DataNotDictLikeError(
                f"Data must be a dict or dict-like object, got {type(data).__name__}"
            )

        row_id = self._key_builder.build(key)
        payload: Optional[Dict[str, Any]] = dict(data) if data else None

        async with self._sessionmaker() as session, session.begin():
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
        async with self._sessionmaker() as session:
            row = await session.get(FSMRow, row_id)
            return cast(Dict[str, Any], row.data or {}) if row else {}

    async def close(self) -> None:  # pragma: no cover
        await self._engine.dispose()
