# -*- coding: utf-8 -*-
"""
文件名：__init__.py
文件描述: app初始化文件
作者: 郑智文
创建日期: 2026/9/10 18:01
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


def create_app() -> FastAPI:
    from app.api import router
    from app.libs.error import register_exception_handlers

    @asynccontextmanager
    async def application_lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 延迟导入配置和模型依赖；构造应用不会启动模型。
        from app.lifespan import lifespan

        async with lifespan(app):
            yield

    app = FastAPI(debug=False, lifespan=application_lifespan)
    # 注册路由
    app.include_router(router, prefix='/api')
    # 注册异常处理
    register_exception_handlers(app)

    return app
