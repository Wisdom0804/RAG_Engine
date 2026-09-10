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
from sqlalchemy import select

from app.libs.enums import ChunkingStrategyEnum
from app.model import dbs
from app.model.parent_chunk import ParentChunk
from tool.rag_engine.chroma_base import ChromaBase
from tool.rag_engine.model_base import model_hub
from tool.rag_engine.parsing_base import document_parser
from tool.rag_engine.splitter_base import Chunk, text_splitter


class RAGBase:
    """
    RAG 引擎：解析→切分→嵌入→入库；query 嵌入→检索→可选 rerank→返回原文。

    父子模式下子块存 chroma、父块存 PG，检索子块后反查 PG 取父块原文。
    非父子模式仅用 chroma。

    职责边界：仅到检索为止，不负责生成；返回上下文原文 list[str]，生成交由上层应用层。
    """

    def __init__(self, collection_name: str = 'learning'):
        self.splitter = text_splitter
        self.parser = document_parser
        self.chroma = ChromaBase(collection_name=collection_name)
        self.model = model_hub

    # ---------- 入库 ----------
    async def ingest(self, path: str, strategy_name: str, **kwargs) -> int:
        """
        解析→切分→嵌入→存库。
        :param path: 文件路径
        :param strategy_name: 切分策略名（fixed_length / semantic / recursive / structural / parent_child）
        :param kwargs: 透传给切分器（parent_child 需传 parent_splitter / child_splitter）
        :return: 入库 chunk 数
        """
        text = await self.parser.parse(path)
        # 语义切分注入 model_hub 复用同一 bge 实例，避免重复加载
        if strategy_name.lower() == 'semantic':
            self.splitter.set_strategy('semantic', embed_fn=self.model.embed_fn, **kwargs)
        else:
            self.splitter.set_strategy(strategy_name, **kwargs)
        chunks = self.splitter.split(text)

        for c in chunks:
            c.metadata['source_file'] = str(path)

        # 父子分流：ParentChildChunking.split 只返回子块，父块需经 .parent 引用收集
        parent_chunks = self._collect_parents(chunks)
        child_chunks = chunks

        if parent_chunks:
            await self._store_parents_pg(parent_chunks)

        # 子块嵌入存 chroma
        docs = [c.document for c in child_chunks]
        embeddings = await self.model.embed(docs)
        for c, emb in zip(child_chunks, embeddings):
            c.embedding = emb
        ids = [str(c.metadata['chunk_id']) for c in child_chunks]
        metas = [c.metadata for c in child_chunks]
        await self.chroma.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
        return len(child_chunks)

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

    async def _store_parents_pg(self, parents: list[Chunk]) -> None:
        """父块批量写入 PG parent_chunk 表"""
        async with dbs.auto_commit() as session:
            for p in parents:
                row = ParentChunk(
                    chunk_id=p.metadata['chunk_id'],
                    document=p.document,
                    metadata_=p.metadata,
                )
                session.add(row)

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
        q_vec = await self.model.embed_query(query)
        resp = await self.chroma.query(q_vec, n_results=top_k, where=where)

        docs = resp.documents
        metas = resp.metadatas

        if rerank:
            ranked = await self.model.rerank(query, docs, top_k)
            # 按 rerank 后的 doc 顺序重排 metas，用初次命中索引定位（避免重复 doc 错位）
            ranked_docs = [d for d, _ in ranked]
            ranked_metas = []
            used = set()
            for d, _ in ranked:
                for i, orig in enumerate(docs):
                    if i not in used and orig == d:
                        ranked_metas.append(metas[i])
                        used.add(i)
                        break
            docs = ranked_docs
            metas = ranked_metas

        # 父子反查 PG（chunk_id 为 bigint，parent_id 从 metadata 直取 int）
        parent_ids = [m.get('parent_id') for m in metas
                      if m.get('parent_id') is not None]
        if parent_ids:
            seen = set()
            uniq_pids = []
            for pid in parent_ids:
                if pid not in seen:
                    seen.add(pid)
                    uniq_pids.append(pid)
            parent_docs = await self._fetch_parents_pg(uniq_pids)
            return parent_docs

        return docs

    async def _fetch_parents_pg(self, parent_ids: list[int]) -> list[str]:
        """按 chunk_id 列表批量查 PG 取父块原文，按 parent_ids 顺序返回"""
        async with dbs.auto_commit() as session:
            rows = (await session.execute(
                select(ParentChunk).where(ParentChunk.chunk_id.in_(parent_ids))
            )).scalars().all()
            # 按 parent_ids 顺序返回，缺失的跳过
            id2doc = {r.chunk_id: r.document for r in rows}
            return [id2doc[pid] for pid in parent_ids if pid in id2doc]


# 模块级单例，与 parser / splitter / chroma_base / model_hub 风格一致
rag_base = RAGBase()


if __name__ == '__main__':
    import asyncio
    from root import ROOT_DIR
    from tool.rag_engine.splitter_base import SplitterFactory

    async def main():
        doc_path = ROOT_DIR / 'docs' / 'file' / '郑智文.pdf'
        query = '郑智文的技术栈与项目经验'

        # ---------- 1) 基础流：语义切分 → 入库 → 检索（无重排） ----------
        # 语义切分不支持 overlap（设计约定：保持语义切分纯净），仅传 chunk_size
        rag = RAGBase(collection_name='learning')
        n = await rag.ingest(str(doc_path), ChunkingStrategyEnum.语义切分.value, chunk_size=300)
        print(f'[semantic] 入库 {n} 条 chunk')

        docs = await rag.retrieve(query, top_k=3)
        print(f'[retrieve 无重排] 命中 {len(docs)} 条:')
        for i, d in enumerate(docs, 1):
            print(f'  {i}. {d[:80].replace(chr(10), " ")}...')

        # ---------- 2) 开启重排 ----------
        ranked = await rag.retrieve(query, top_k=3, rerank=True)
        print(f'[retrieve 重排] 命中 {len(ranked)} 条:')
        for i, d in enumerate(ranked, 1):
            print(f'  {i}. {d[:80].replace(chr(10), " ")}...')

        # ---------- 3) 父子模式：父块存 PG、子块存 chroma，检索子块反查父块 ----------
        # 用独立 collection 避免与上面定长 chunk 混检
        rag_pc = RAGBase(collection_name='parent_child_test')
        parent = SplitterFactory.create_strategy(ChunkingStrategyEnum.结构切分.value, chunk_size=800, overlap=0)
        child = SplitterFactory.create_strategy(ChunkingStrategyEnum.语义切分.value, chunk_size=200)
        n2 = await rag_pc.ingest(str(doc_path), 'parent_child',
                                 parent_splitter=parent, child_splitter=child)
        print(f'[parent_child] 入库 {n2} 条子 chunk')

        p_docs = await rag_pc.retrieve(query, top_k=3, rerank=True,
                                       where={'source_file': str(doc_path)})
        print(f'[parent_child retrieve] 命中 {len(p_docs)} 条父块:')
        for i, d in enumerate(p_docs, 1):
            print(f'  {i}. {d[:80].replace(chr(10), " ")}...')

    asyncio.run(main())
