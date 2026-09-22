# -*- coding: utf-8 -*-
"""
文件名：rag_base.py
文件描述: RAG 引擎基类，串起解析→切分→嵌入→入库与 query 嵌入→检索→可选 rerank→返回原文
作者: 郑智文
创建日期: 2026/9/10
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

try:
    from tool.rag_engine.chroma_base import ChromaBase, CollectionConflict
except ImportError:  # isolated legacy test doubles
    from tool.rag_engine.chroma_base import ChromaBase
    class CollectionConflict(ValueError):
        pass
from tool.rag_engine.model_base import ModelHub, ModelNotReady
from tool.rag_engine.parsing_base import DocumentParser
from tool.rag_engine.splitter_base import Chunk, SplitterFactory, ChunkingStrategy

if TYPE_CHECKING:
    from app.model.parent_chunk import ParentChunkStore


class StrategyUnavailable(ValueError):
    """当前资源不支持所选策略。"""


class RAGBase:
    """
    RAG 引擎：解析→切分→嵌入→入库；query 嵌入→检索→可选 rerank→返回原文。

    父子模式下子块存 chroma、父块存 PG，检索子块后反查 PG 取父块原文。
    非父子模式仅用 chroma。

    职责边界：仅到检索为止，不负责生成；返回上下文原文 list[str]，生成交由上层应用层。
    """

    def __init__(self, *, chroma: ChromaBase, parser: DocumentParser, model: ModelHub,
                 next_id: Callable[[], int], parents: "ParentChunkStore | None" = None):
        self.parser = parser
        self.chroma = chroma
        self.model = model
        self.next_id = next_id
        self.parents = parents

    # ---------- 入库 ----------
    async def ingest(self, path: str, strategy_name: str, *, source_file: str | None = None,
                     file_sha256: str | None = None, **kwargs) -> int:
        """解析、切分、嵌入并存储文档。

        Args:
            path: 待解析的本地文件路径。
            strategy_name: 已注册的切分策略名。
            source_file: 对外使用的来源文件名；省略时兼容本地路径调用。
            file_sha256: 上传文件内容指纹，供重复入库检测。
            **kwargs: 传给切分器的完整配置。

        Returns:
            存入向量库的块数，父子模式为子块数。

        Raises:
            ValueError: 文档为空、无法切分或嵌入数量不匹配。
        """
        if file_sha256 is not None:
            await self.chroma.check_strategy(strategy_name)
            if (await self.chroma.get(where={'file_sha256': file_sha256}, limit=1)).ids:
                raise CollectionConflict('同一集合中不能重复上传同一文件')
        self.model.require_ready()
        strategy = self._create_strategy(strategy_name, **kwargs)
        if file_sha256 is not None:
            await self.chroma.bind_strategy(strategy_name)
        text = await self.parser.parse(path)
        if not text.strip():
            raise ValueError('文档没有可检索文本')
        chunks = await asyncio.to_thread(strategy.split, text)
        if not chunks:
            raise ValueError('文档没有可检索文本块')

        # 父子分流：ParentChildChunking.split 只返回子块，父块需经 .parent 引用收集
        parent_chunks = self._collect_parents(chunks)
        child_chunks = chunks
        for c in child_chunks + parent_chunks:
            c.metadata['source_file'] = source_file if source_file is not None else str(path)
            if file_sha256 is not None:
                c.metadata['file_sha256'] = file_sha256

        # 子块嵌入存 chroma
        docs = [c.document for c in child_chunks]
        embeddings = await self.model.embed(docs)
        if len(embeddings) != len(child_chunks):
            raise ValueError('嵌入数量与文本块数量不匹配')
        ids = [str(c.metadata['chunk_id']) for c in child_chunks]
        metas = [c.metadata for c in child_chunks]
        try:
            if parent_chunks:
                await self.parents.add(parent_chunks)
            await self.chroma.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
        except Exception:
            # 两种存储不支持共同事务；异常时按本次生成的 ID 补偿清理。
            try:
                await self.chroma.delete(ids)
            finally:
                if parent_chunks:
                    await self.parents.delete([c.metadata['chunk_id'] for c in parent_chunks])
            raise
        return len(child_chunks)

    def _create_strategy(self, name: str, **kwargs) -> ChunkingStrategy:
        """每次入库独立构造轻量策略，模型与 ID 生成器继续共享。"""
        name = name.lower()
        if name == 'parent_child':
            if self.parents is None:
                raise StrategyUnavailable('父子切分需要配置 PostgreSQL 父块存储')
            if 'parent_splitter' not in kwargs:
                kwargs['parent_splitter'] = SplitterFactory.create_strategy(
                    'structural', chunk_size=kwargs.pop('parent_chunk_size', 800),
                    overlap=0, next_id=self.next_id)
            if 'child_splitter' not in kwargs:
                kwargs['child_splitter'] = SplitterFactory.create_strategy(
                    'recursive', chunk_size=kwargs.pop('chunk_size', 200),
                    overlap=kwargs.pop('overlap', 0), next_id=self.next_id)
        else:
            kwargs['next_id'] = self.next_id
            if name == 'semantic':
                kwargs['embed_fn'] = self.model.embed_fn
                if kwargs['embed_fn'] is None:
                    raise StrategyUnavailable('语义切分需要本地嵌入模型')
        return SplitterFactory.create_strategy(name, **kwargs)

    def _collect_parents(self, chunks: list[Chunk]) -> list[Chunk]:
        """从子块 .parent 引用去重收集父块（仅父子策略产生 parent）"""
        seen = set()
        parents = []
        for c in chunks:
            p = c.parent
            if p is None:
                continue
            pid = p.metadata.get('chunk_id')
            if pid in seen:
                continue
            seen.add(pid)
            parents.append(p)
        return parents

    # ---------- 检索 ----------
    async def retrieve(self, query: str, top_k: int = 5,
                       rerank: bool = False, where: dict = None) -> list[str]:
        """
        query 嵌入→chroma 查→可选 rerank→父子反查 PG→返回原文 list[str]。

        :param query: 检索 query
        :param top_k: 检索条数；父子去重后最终返回可能 < top_k
        :param rerank: 是否对 chroma 初检结果重排
        :param where: 按 metadata 过滤（如 {'source_file': ...}）
        :return: 父子模式返回父块原文（按 parent_id 去重保序）；非父子返回 chunk 原文
        """
        self.model.require_ready()
        q_vec = await self.model.embed_query(query)
        resp = await self.chroma.query(q_vec, n_results=top_k, where=where)

        docs = resp.documents
        metas = resp.metadatas

        if not docs:
            return []

        if rerank:
            ranked = await self.model.rerank(query, docs, top_k)
            # 候选索引同时定位文本和 metadata，重复文本也保留身份。
            docs = [docs[i] for i, _ in ranked]
            metas = [metas[i] for i, _ in ranked]

        # 父子反查 PG（chunk_id 为 bigint，parent_id 从 metadata 直取 int）
        parent_ids = [m.get('parent_id') for m in metas
                      if m.get('parent_id') is not None]
        if parent_ids:
            uniq_pids = list(dict.fromkeys(parent_ids))
            if self.parents is None:
                raise StrategyUnavailable('父子检索需要配置 PostgreSQL 父块存储')
            return await self.parents.fetch(uniq_pids)

        return docs


if __name__ == '__main__':
    from app.lifespan import lifespan
    from root import ROOT_DIR

    async def main():
        # CLI 与 HTTP 共享显式装配；默认示例只返回上下文，不生成答案。
        async with lifespan(None) as service:
            rag = await service._get_rag('learning', create=True)
            path = ROOT_DIR / 'docs' / 'file' / '郑智文.pdf'
            count = await rag.ingest(str(path), 'semantic', chunk_size=300)
            print(f'入库 {count} 个块')
            print(await rag.retrieve('郑智文的技术栈与项目经验', top_k=3, rerank=True))
            rag_pc = await service._get_rag('parent_child_test', create=True)
            await rag_pc.ingest(str(path), 'parent_child')
            print(await rag_pc.retrieve('郑智文的技术栈与项目经验', top_k=3))

    asyncio.run(main())
