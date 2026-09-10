# -*- coding: utf-8 -*-
"""
文件名：model_base.py
文件描述: 模型接入中心，统一管理嵌入（embedding）与重排（rerank）模型，本地/API 双轨可切
作者: 郑智文
创建日期: 2026/9/10
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
import asyncio

import dashscope
from http import HTTPStatus
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config.security import secure

# 阿里华北2（北京）地域的 maas 入口（与 TPA/ALi/qwen 测试脚本一致）
dashscope.base_http_api_url = "https://llm-udh4a0m8ljmmeirn.cn-beijing.maas.aliyuncs.com/api/v1"

# 阿里 qwen3.7 系列模型名
_QWEN_EMBED_MODEL = "qwen3.7-text-embedding-flash"
_QWEN_RERANK_MODEL = "qwen3.7-text-rerank"
# qwen text-embedding 单次最多 20 条
_QWEN_EMBED_BATCH = 20
# 指定向量维度（仅 qwen3.7-text-embedding / v3 / v4 支持）
_QWEN_EMBED_DIMENSION = 1024


class ModelHub:
    """
    模型接入中心：统一管理嵌入与重排模型，本地/API 双轨可切。

    - 嵌入默认本地 BAAI/bge-small-zh-v1.5，secure.EMBEDDING_NAME 命中 qwen 前缀则走 API
    - 重排默认本地 cross-encoder/ms-marco-MiniLM-L6-v2，secure.RERANK_NAME 命中 qwen 前缀则走 API
    - embed_fn 属性供 SemanticChunking.configure(embed_fn=...) 注入复用同一 bge 实例（仅本地模式）
    """

    # 阿里模型名标识（命中则走 API）
    _QWEN_MARK = 'qwen'

    def __init__(self):
        self._embedder = None
        self._reranker = None
        self._init_embedder()
        self._init_reranker()

    # ---------- 嵌入 ----------
    def _init_embedder(self):
        name = getattr(secure, 'EMBEDDING_NAME', '') or ''
        if name.lower().startswith(self._QWEN_MARK):
            self._embed_mode = 'api'
            self._embed_model_name = name
        else:
            self._embed_mode = 'local'
            self._embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            self._embed_model_name = "BAAI/bge-small-zh-v1.5"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量编码文本为向量。
        本地走同步 encode 包 to_thread；API 走 dashscope，按 batch<=20 分批。
        """
        if self._embed_mode == 'local':
            vecs = await asyncio.to_thread(self._embedder.encode, texts)
            return vecs.tolist()
        return await self._embed_api(texts)

    async def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """阿里 qwen3.7-text-embedding-flash，按 batch<=20 分批"""
        results: list[list[float]] = []
        for i in range(0, len(texts), _QWEN_EMBED_BATCH):
            batch = texts[i:i + _QWEN_EMBED_BATCH]
            resp = await asyncio.to_thread(
                dashscope.TextEmbedding.call,
                api_key=secure.QWEN_API_KEY,
                model=_QWEN_EMBED_MODEL,
                input=batch,
                dimension=_QWEN_EMBED_DIMENSION,
                text_type="document",  # 入库/聚类/分类用 document，检索 query 用 query
                output_type="dense",
            )
            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(f"qwen embedding 调用失败: {resp}")
            # output.embeddings 按 text_index 升序，与输入顺序一致
            for item in resp["output"]["embeddings"]:
                results.append(item["embedding"])
        return results

    async def embed_query(self, text: str) -> list[list[float]]:
        """检索 query 编码，API 模式 text_type=query"""
        if self._embed_mode == 'local':
            vecs = await asyncio.to_thread(self._embedder.encode, [text])
            return vecs.tolist()
        resp = await asyncio.to_thread(
            dashscope.TextEmbedding.call,
            api_key=secure.QWEN_API_KEY,
            model=_QWEN_EMBED_MODEL,
            input=text,
            dimension=_QWEN_EMBED_DIMENSION,
            text_type="query",
            output_type="dense",
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"qwen embedding(query) 调用失败: {resp}")
        return [resp["output"]["embeddings"][0]["embedding"]]

    @property
    def embed_fn(self):
        """供 SemanticChunking.configure(embed_fn=...) 注入复用（仅本地模式可注入）"""
        if self._embed_mode != 'local':
            raise ValueError("切分器注入仅支持本地嵌入模型")
        return self._embedder

    # ---------- 重排 ----------
    def _init_reranker(self):
        name = getattr(secure, 'RERANK_NAME', '') or ''
        if name.lower().startswith(self._QWEN_MARK):
            self._rerank_mode = 'api'
            self._rerank_model_name = name
        else:
            self._rerank_mode = 'local'
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
            self._rerank_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"

    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[tuple[str, float]]:
        """
        重排，返回 (doc, score) 列表降序，截 top_k。
        """
        if self._rerank_mode == 'local':
            pairs = [(query, d) for d in docs]
            scores = await asyncio.to_thread(self._reranker.predict, pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        return await self._rerank_api(query, docs, top_k)

    async def _rerank_api(self, query: str, docs: list[str], top_k: int) -> list[tuple[str, float]]:
        """阿里 qwen3.7-text-rerank"""
        resp = await asyncio.to_thread(
            dashscope.TextReRank.call,
            api_key=secure.QWEN_API_KEY,
            model=_QWEN_RERANK_MODEL,
            query=query,
            documents=docs,
            top_n=top_k,
            return_documents=False,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"qwen rerank 调用失败: {resp}")
        # output.results 按 relevance_score 降序，每项含 index（对应输入 docs 下标）
        ranked: list[tuple[str, float]] = []
        for item in resp["output"]["results"]:
            idx = item["index"]
            ranked.append((docs[idx], float(item["relevance_score"])))
        return ranked[:top_k]


# 模块级单例，与 document_parser / text_splitter / chroma_base 风格一致
model_hub = ModelHub()
