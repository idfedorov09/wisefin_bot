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
from sqlalchemy import String, DateTime, Table, func, inspect
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
    CREATE = "create"
    CREATE_DROP = "create-drop"
    VALIDATE = "validate"
    UPDATE = "update"


class SchemaError(RuntimeError):
    """Base class for schema related errors."""


class SchemaValidationError(SchemaError):
    """Raised when database schema validation fails."""


class SchemaUpdateError(SchemaError):
    """Raised when database schema auto-update fails."""


_Event = Literal["start", "stop"]

_default_schema_mode = SchemaMode(
    os.getenv("AIOGRAM_SQLALCHEMY_STORAGE_SCHEMA_MODE", SchemaMode.UPDATE.value)
)


class BaseSchemaModeStrategy(ABC):
    def __init__(self, table: Table) -> None:
        self._table = table

    @abstractmethod
    async def on_start(self, engine: AsyncEngine) -> None:
        raise NotImplementedError

    @abstractmethod
    async def on_stop(self, engine: AsyncEngine) -> None:
        raise NotImplementedError


class NoneSchemaModeStrategy(BaseSchemaModeStrategy):
    async def on_start(self, engine: AsyncEngine) -> None:
        pass

    async def on_stop(self, engine: AsyncEngine) -> None:
        pass


class CreateSchemaModeStrategy(BaseSchemaModeStrategy):
    async def on_start(self, engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(self._recreate_table)

    async def on_stop(self, engine: AsyncEngine) -> None:
        pass

    def _recreate_table(self, sync_conn) -> None:
        self._table.drop(sync_conn, checkfirst=True)
        self._table.create(sync_conn, checkfirst=True)


class CreateDropSchemaModeStrategy(BaseSchemaModeStrategy):
    async def on_start(self, engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(self._create_table)

    async def on_stop(self, engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(self._drop_table)

    def _create_table(self, sync_conn) -> None:
        self._table.create(sync_conn, checkfirst=True)

    def _drop_table(self, sync_conn) -> None:
        self._table.drop(sync_conn, checkfirst=True)


class ValidateSchemaModeStrategy(BaseSchemaModeStrategy):
    async def on_start(self, engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(self._validate_table)

    async def on_stop(self, engine: AsyncEngine) -> None:
        pass

    def _validate_table(self, sync_conn) -> None:
        inspector = inspect(sync_conn)
        table_name = self._table.name

        if not inspector.has_table(table_name):
            raise SchemaValidationError(
                f"Table `{table_name}` is missing; run with schema_mode='create' or 'update' first."
            )

        actual_columns = {col["name"]: col for col in inspector.get_columns(table_name)}
        expected_columns = {column.name: column for column in self._table.columns}

        missing_columns = sorted(set(expected_columns) - set(actual_columns))
        unexpected_columns = sorted(set(actual_columns) - set(expected_columns))
        if missing_columns:
            raise SchemaValidationError(
                f"Missing columns in `{table_name}`: {', '.join(missing_columns)}"
            )
        if unexpected_columns:
            raise SchemaValidationError(
                f"Unexpected columns in `{table_name}`: {', '.join(unexpected_columns)}"
            )

        dialect = inspector.bind.dialect
        for column_name, expected in expected_columns.items():
            actual = actual_columns[column_name]
            expected_type = expected.type.compile(dialect=dialect).lower()
            actual_type = actual["type"].compile(dialect=dialect).lower()
            if expected_type != actual_type:
                raise SchemaValidationError(
                    f"Column `{column_name}` type mismatch: expected {expected_type}, got {actual_type}"
                )

            expected_nullable = bool(expected.nullable)
            actual_nullable = bool(actual.get("nullable", True))
            if expected_nullable != actual_nullable:
                raise SchemaValidationError(
                    f"Column `{column_name}` nullable mismatch: expected {expected_nullable}, got {actual_nullable}"
                )

            expected_default = expected.server_default is not None
            actual_default = actual.get("default") is not None
            if expected_default != actual_default:
                raise SchemaValidationError(
                    f"Column `{column_name}` server default presence mismatch."
                )

        expected_pk = tuple(column.name for column in self._table.primary_key.columns)
        pk_info = inspector.get_pk_constraint(table_name) or {}
        actual_pk = tuple(pk_info.get("constrained_columns") or ())
        if expected_pk != actual_pk:
            raise SchemaValidationError(
                f"Primary key mismatch: expected {expected_pk or 'none'}, got {actual_pk or 'none'}"
            )


class UpdateSchemaModeStrategy(BaseSchemaModeStrategy):
    async def on_start(self, engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(self._auto_upgrade)

    async def on_stop(self, engine: AsyncEngine) -> None:
        pass

    def _auto_upgrade(self, sync_conn) -> None:
        try:
            from alembic.autogenerate import api as alembic_autogen
            from alembic.migration import MigrationContext
            from alembic.operations import BatchOperations, Operations
            from alembic.operations import ops as alembic_ops
        except ImportError as exc:  # pragma: no cover
            raise SchemaUpdateError(
                "Alembic is required for schema_mode='update'. Install alembic to enable auto-migrations."
            ) from exc

        context = MigrationContext.configure(
            sync_conn,
            opts={
                "target_metadata": Base.metadata,
                "compare_type": True,
                "compare_server_default": True,
                "render_as_batch": True,
            },
        )
        migration_script = alembic_autogen.produce_migrations(context, Base.metadata)
        upgrade_ops = migration_script.upgrade_ops
        if upgrade_ops.is_empty():
            return

        operations = Operations(context)

        def targets_fsm(operation: alembic_ops.MigrateOperation) -> bool:
            table_name = getattr(operation, "table_name", None)
            if table_name is not None:
                return table_name == self._table.name
            if isinstance(operation, alembic_ops.OpContainer):
                return any(targets_fsm(op) for op in operation.ops)
            return False

        def apply_ops(
                container: alembic_ops.OpContainer,
                executor: Union[Operations, BatchOperations],
        ) -> None:
            for operation in container.ops:
                if not targets_fsm(operation):
                    continue
                if isinstance(operation, alembic_ops.ModifyTableOps):
                    if not hasattr(executor, "batch_alter_table"):
                        raise SchemaUpdateError(
                            f"Nested batch operations are not supported for table `{self._table.name}`."
                        )
                    with executor.batch_alter_table(
                            operation.table_name,
                            schema=operation.schema,
                    ) as batch_op:
                        apply_ops(operation, batch_op)
                elif isinstance(operation, alembic_ops.OpContainer):
                    apply_ops(operation, executor)
                else:
                    executor.invoke(operation)

        with context.begin_transaction():
            apply_ops(upgrade_ops, operations)


class SchemaModeEventContext:
    _STRATEGY_MAP: dict[SchemaMode, type[BaseSchemaModeStrategy]] = {
        SchemaMode.NONE: NoneSchemaModeStrategy,
        SchemaMode.CREATE: CreateSchemaModeStrategy,
        SchemaMode.CREATE_DROP: CreateDropSchemaModeStrategy,
        SchemaMode.VALIDATE: ValidateSchemaModeStrategy,
        SchemaMode.UPDATE: UpdateSchemaModeStrategy,
    }

    def __init__(self, schema_mode: SchemaMode, engine: AsyncEngine, table: Table) -> None:
        self._table = table
        self._strategy: BaseSchemaModeStrategy = self._STRATEGY_MAP[schema_mode](self._table)
        self._engine: AsyncEngine = engine

    def set_table(self, table: Table) -> "SchemaModeEventContext":
        self._table = table
        return self

    def set_strategy(self, strategy: SchemaMode) -> "SchemaModeEventContext":
        self._strategy = self._STRATEGY_MAP[strategy](self._table)
        return self

    def set_engine(self, engine: AsyncEngine) -> "SchemaModeEventContext":
        self._engine = engine
        return self

    async def event(self, event: _Event) -> "SchemaModeEventContext":
        if event == "start":
            await self._strategy.on_start(self._engine)
        elif event == "stop":
            await self._strategy.on_stop(self._engine)
        return self


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
        """Возвращает новый контекстный менеджер"""
        raise NotImplementedError

    @abstractmethod
    async def __aenter__(self) -> AsyncSession:
        """Вход в контекст, возвращает асинхронную сессию."""
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Выход из контекста"""
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
        # TODO: траблы с table_name
        if table_name != FSMRow.__tablename__:
            FSMRow.__tablename__ = table_name  # type: ignore[attr-defined]
        if isinstance(schema_mode, str):
            schema_mode = SchemaMode(schema_mode)

        self._key_builder: KeyBuilder = key_builder or DefaultKeyBuilder()
        self._engine: AsyncEngine = engine
        self._session_context = session_context \
                                or SQLAlchemyStorage._make_simple_session_context(
            self._engine,
            sessionmaker_kwargs,
            use_transaction
        )
        self._schema_mode = schema_mode or SchemaMode.NONE
        self._schema_mode_event_ctx = SchemaModeEventContext(
            schema_mode=self._schema_mode,
            engine=self._engine,
            table=cast(Table, FSMRow.__table__),
        )
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
        await storage.schema_event("start") # TODO: а если экземпляр создается через __init__ ?
        return storage

    async def schema_event(self, event: _Event) -> None:
        await self._schema_mode_event_ctx.event(event)

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
