# -*- coding: utf-8 -*-
"""
文件名：__init__.py.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/3 14:43
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from fastapi import APIRouter

from app.api.test import router as test_router
from app.api.rag.rag import router as rag_router

router = APIRouter()
router.include_router(test_router)
router.include_router(rag_router)
