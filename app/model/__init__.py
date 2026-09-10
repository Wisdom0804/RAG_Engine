# -*- coding: utf-8 -*-
"""
文件名：__init__.py.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/3 14:49
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""


from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config.security import secure


class Database:
    # 异步数据库封装，全局复用引擎与会话工厂
    def __init__(self, url: str = secure.POSTGRE_DATABASE_URL):
        self.engine = create_async_engine(
            url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            future=True,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=30,
        )
        self.async_session = sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self.Base = declarative_base()

    async def get_session(self):
        # FastAPI依赖注入用，自动提交与回滚
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def auto_commit(self):
        # 手动控制事务的上下文
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

dbs = Database(secure.POSTGRE_DATABASE_URL)