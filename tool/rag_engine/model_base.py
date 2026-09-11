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

from app.config.security import secure

# qwen text-embedding 单次最多 20 条
_QWEN_EMBED_BATCH = 20
# 指定向量维度（仅 qwen3.7-text-embedding / v3 / v4 支持）
_QWEN_EMBED_DIMENSION = 1024


class ModelHub:
    """
    模型接入中心：统一管理嵌入与重排模型，本地/API 双轨可切。

    - 模式分别取 secure.EMBEDDING_MODE / RERANK_MODE，不按模型名前缀推断。
    - 本地模型由外部生命周期加载后挂载，本类不构造模型。
    - embed_fn 本地模式返回共享实例，API 模式返回 None；语义切分不隐式兜底。
    """

    def __init__(self) -> None:
        self.unload()

    def unload(self) -> None:
        """释放模型引用，保留单例对象身份。"""
        self.ready = False
        self._embedder = None
        self._reranker = None
        self._embed_mode = None
        self._rerank_mode = None
        self._embed_model_name = None
        self._rerank_model_name = None

    def mount(self, *, embedder=None, reranker=None) -> None:
        """验证完成后一次挂载已加载实例，失败不留下半就绪状态。"""
        if self.ready:
            raise RuntimeError('ModelHub 已挂载模型')
        secure.validate_models()
        if secure.EMBEDDING_MODE == 'local' and not callable(getattr(embedder, 'encode', None)):
            raise ValueError('本地嵌入模式需要已加载的 embedder')
        if secure.RERANK_MODE == 'local' and not callable(getattr(reranker, 'predict', None)):
            raise ValueError('本地重排模式需要已加载的 reranker')
        self._embed_mode = secure.EMBEDDING_MODE
        self._rerank_mode = secure.RERANK_MODE
        self._embedder = embedder if self._embed_mode == 'local' else None
        self._reranker = reranker if self._rerank_mode == 'local' else None
        self._embed_model_name = (secure.LOCAL_EMBEDDING_NAME if self._embed_mode == 'local'
                                  else secure.QWEN_EMBEDDING_NAME)
        self._rerank_model_name = (secure.LOCAL_RERANK_NAME if self._rerank_mode == 'local'
                                   else secure.QWEN_RERANK_NAME)
        if 'api' in (self._embed_mode, self._rerank_mode):
            dashscope.base_http_api_url = secure.QWEN_API_BASE
        self.ready = True

    def require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError('ModelHub 未就绪，请先通过生命周期加载并挂载模型')

    # ---------- 嵌入 ----------
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量编码文本为向量。
        本地走同步 encode 包 to_thread；API 走 dashscope，按 batch<=20 分批。
        """
        self.require_ready()
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
                model=self._embed_model_name,
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
        self.require_ready()
        if self._embed_mode == 'local':
            vecs = await asyncio.to_thread(self._embedder.encode, [text])
            return vecs.tolist()
        resp = await asyncio.to_thread(
            dashscope.TextEmbedding.call,
            api_key=secure.QWEN_API_KEY,
            model=self._embed_model_name,
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
        """供 SemanticChunking.configure(embed_fn=...) 注入复用同一 bge 实例。

        - 本地模式：返回已加载的 SentenceTransformer 实例，复用避免重复加载
        - API 模式：返回 None，语义切分需另行显式注入支持同步 encode 的实例。
        """
        self.require_ready()
        if self._embed_mode != 'local':
            return None
        return self._embedder

    # ---------- 重排 ----------
    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[tuple[str, float]]:
        """
        重排，返回 (doc, score) 列表降序，截 top_k。
        """
        self.require_ready()
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
            model=self._rerank_model_name,
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
