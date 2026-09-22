"""显式创建 PostgreSQL 引擎；ORM 定义不依赖数据库实例。"""
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Database:
    def __init__(self, url: str):
        self.engine = create_async_engine(
            url, echo=False, pool_size=10, max_overflow=20,
            pool_pre_ping=True, pool_recycle=3600, pool_timeout=30,
        )
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def auto_commit(self):
        """由 SQLAlchemy 管理成功提交、异常回滚和关闭。"""
        async with self.async_session.begin() as session:
            yield session
