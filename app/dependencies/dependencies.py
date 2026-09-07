# -*- coding: utf-8 -*-
"""
文件名：dependencies.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/4 15:18
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from fastapi import Request
from tool.rag_engine.chroma_base import ChromaBase


async def get_chroma(request: Request) -> ChromaBase:
    """
    依赖注入方式获取 RedisHandler
    :param request:
    :return:
    """
    return request.app.state.redis_handler