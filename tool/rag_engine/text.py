# -*- coding: utf-8 -*-
"""
文件名：chroma_base.py
文件描述:
作者: 郑智文
创建日期: 2026/9/7 10:02
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from uuid import uuid4
import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
from . import chromadb_client


class TestBase:
    """
    向量数据库操作基类，单例模式，提供基础的CRUD
    """

    def __init__(self, collection_name: str = 'learning'):
        """
        磁盘持久化存储，单例模式不需要连接池
        """

        # 默认嵌入模型
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # 默认重排模型
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        # 集合
        self.collection = chromadb_client.get_or_create_collection(name=collection_name)

    async def embed_chunk(self, chunk: str) -> list[float]:
        """
        向量化切片
        """
        embedding = self.embedding_model.encode(chunk)
        return embedding.tolist()

    async def add(self, chunks: list[str] | None = None, embeddings: list[str] | None = None,
                  metadatas: list[dict] | None = None):
        """
        添加
        """
        param_info = []
        chunk_len = len(chunks) if chunks else 0
        embedding_len = len(embeddings) if embeddings else 0
        metadata_len = len(metadatas) if metadatas else 0

        if chunks is not None:
            param_info.append(("chunks", chunk_len))
        if embeddings is not None:
            param_info.append(("embeddings", embedding_len))
        if metadatas is not None:
            param_info.append(("metadatas", metadata_len))

        # 只要有至少一个传入参数，全部长度必须相等
        if len(param_info) > 1:
            first_name, first_len = param_info[0]
            for name, length in param_info[1:]:
                if length != first_len:
                    raise ValueError(
                        f"参数长度不匹配：{first_name}长度={first_len}，{name}长度={length}。"
                        "chunks、embeddings、metadatas 同时传入任意两个或全部时，数组长度必须保持一致"
                    )

        data_from = dict()

        if chunk_len:
            data_from["documents"] = chunks
        if embedding_len:
            data_from["embeddings"] = embeddings
        if metadata_len:
            data_from["metadatas"] = metadatas

        ids = [str(uuid4()) for _ in range(max(chunk_len, embedding_len, metadata_len))]
        data_from["ids"] = ids

        self.collection.add(**data_from)

    async def retrieve(self, query: str, top_k: int) -> list[str]:
        """
        检索
        """
        query_embedding = await self.embed_chunk(query)
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return res['documents'][0]

    async def rerank(self, query: str, retrieved_chunks: list[str], top_k: int) -> list[str]:
        """
        重排函数
        """
        pairs = [
            (query, chunk)
            for chunk in retrieved_chunks
        ]
        scores = self.cross_encoder.predict(pairs)
        chunk_with_score_list = [
            (chunk, score)
            for chunk, score in zip(retrieved_chunks, scores)
        ]
        chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)

        return [chunk for chunk, score in chunk_with_score_list[:top_k]]


if __name__ == '__main__':
    chroma = ChromaBase()
    asyncio.run(chroma.add(chunks=['Hello', '扣你吉瓦', 'ohhhhhhh']))
    # asyncio.run(chroma.retrieve('你好',2))
    res = asyncio.run(chroma.rerank('你好', ['Hello', '扣你吉瓦', 'ohhhhhhh'], 2))
    print(res)