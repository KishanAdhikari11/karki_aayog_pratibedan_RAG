import contextlib
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    AsyncConnection,
    AsyncEngine,
    async_sessionmaker,
)
from utils import get_logger
from config import config
from models import Base
from sqlalchemy import text

logger = get_logger()


class DatabaseSessionManager:
    def __init__(self):
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker | None = None

    def init(self):
        self._engine = create_async_engine(config.DB_CONFIG, echo=False)
        self._sessionmaker = async_sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    async def close(self):
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
        else:
            raise RuntimeError("Database engine is not initialized.")

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        if not self._engine:
            raise RuntimeError("Database engine is not initialized.")
        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception as e:
                await connection.rollback()
                logger.error(f"Error during database connection: {e}")
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_all(self, connection: AsyncConnection):
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await connection.run_sync(Base.metadata.create_all)

    async def drop_all(self, connection: AsyncConnection):
        await connection.run_sync(Base.metadata.drop_all)


sessionmanager = DatabaseSessionManager()


async def get_db():
    async with sessionmanager.session() as session:
        yield session
