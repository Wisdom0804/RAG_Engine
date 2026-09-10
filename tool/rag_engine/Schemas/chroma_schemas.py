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


class ChromaQueryResponse(BaseModel):
    """
    ChromaBase.query 的响应，含 metadata 与距离，供 rerank 与 parent_id 溯源。
    query 按 query_embeddings 检索，ChromaDB 原始返回各字段均为外层 list[list]，
    此处只取第一组（单条 query）的结果展平为一维。
    """
    ids: list[str] = Field(..., description="命中的记录 ID 列表")
    documents: list[str] = Field(..., description="命中的文档内容列表")
    metadatas: list[dict] = Field(..., description="命中的元数据列表，供 parent_id 溯源")
    distances: list[float] = Field(..., description="命中的距离分数列表，越小越相似")
