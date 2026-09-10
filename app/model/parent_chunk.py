# -*- coding: utf-8 -*-
"""
文件名：parent_chunk.py
文件描述: 父子切分策略的父块持久化 ORM 模型
作者: 郑智文
创建日期: 2026/9/10
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from sqlalchemy import JSON, BigInteger, Column, DateTime, Text, func

from app.model import dbs


class ParentChunk(dbs.Base):
    """
    父子切分策略的父块持久化表，子块存 chroma，父块存此表。

    检索阶段：子块在 chroma 命中后，按 metadata['parent_id'] 反查本表取父块原文，
    返回更完整的上下文（Parent Document Retrieval 思路）。
    """
    __tablename__ = 'parent_chunk'
    __table_args__ = {'schema': 'rag_engine'}

    chunk_id = Column(BigInteger, primary_key=True, nullable=False, comment='雪花ID')
    document = Column(Text, comment='原文')
    metadata_ = Column('metadata', JSON, comment='元数据')
    create_time = Column(DateTime(timezone=True), server_default=func.now(), comment='入库事件')
