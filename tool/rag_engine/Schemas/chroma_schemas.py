# -*- coding: utf-8 -*-
"""
文件名：chroma_schemas.py
文件描述: ChromaBase 的响应模型
作者: 郑智文
创建日期: 2026/9/7 11:50
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from pydantic import BaseModel, Field


class ChromaGetResponse(BaseModel):
    """
    ChromaBase.get 的响应类型。仅保留 id、documents、embeddings、metadatas 四个字段，
    丢弃 ChromaDB 原始响应中的 uris、included、data 等无关字段。
    """
    ids: list[str] = Field(..., description="记录 ID 列表")
    documents: list[str] | None = Field(default=None, description="文档内容列表，未 include 时为 None")
    embeddings: list[list[float]] | None = Field(default=None, description="向量列表，未 include 时为 None")
    metadatas: list[dict] | None = Field(default=None, description="元数据列表，未 include 时为 None")
