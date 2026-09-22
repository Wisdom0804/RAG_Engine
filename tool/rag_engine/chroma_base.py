# -*- coding: utf-8 -*-
"""
文件名：chroma_base.py
文件描述: 向量数据库操作基类，仅提供基础的增删改查（CRUD）
作者: 郑智文
创建日期: 2026/9/5 10:04
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
import asyncio
from uuid import uuid4

from chromadb.api import ClientAPI
from tool.rag_engine.Schemas.chroma_schemas import ChromaGetResponse, ChromaQueryResponse


class CollectionConflict(ValueError):
    """集合策略或文件内容与已有数据冲突。"""


class ChromaBase:
    """
    向量数据库操作基类，提供基础的 CRUD。

    阻塞的 ChromaDB 调用通过 asyncio.to_thread 丢到线程池，避免阻塞事件循环。
    嵌入向量由调用方外部计算后传入，本类不负责 embedding / rerank。
    """

    def __init__(self, client: ClientAPI, collection_name: str = 'learning', *, create: bool = True):
        self.client = client
        self.collection = (self.client.get_or_create_collection(name=collection_name)
                           if create else self.client.get_collection(name=collection_name))

    async def check_strategy(self, strategy: str) -> None:
        """只检查策略一致性；绑定必须在文件和模型就绪后执行。"""
        current = (self.collection.metadata or {}).get('rag_strategy')
        if current is not None and current != strategy:
            raise CollectionConflict('集合已绑定其他切分策略')
        if current is None and await asyncio.to_thread(self.collection.count):
            raise CollectionConflict('已有集合缺少策略标记，请使用新集合')

    async def bind_strategy(self, strategy: str) -> None:
        """首次合法入库前持久绑定策略，保留其他集合元数据。"""
        metadata = dict(self.collection.metadata or {})
        if metadata.get('rag_strategy') is None:
            await asyncio.to_thread(self.collection.modify, metadata={**metadata, 'rag_strategy': strategy})

    async def add(self, documents: list[str] | None = None,
                  embeddings: list[list[float]] | None = None,
                  metadatas: list[dict] | None = None,
                  ids: list[str] | None = None) -> list[str]:
        """
        新增数据（Create）。documents / embeddings / metadatas 任选传入，
        同时传入多个时长度必须一致。未传 ids 时自动生成 UUID。
        :return: 写入的 ids 列表
        """
        lengths = [len(x) for x in (documents, embeddings, metadatas, ids) if x is not None]
        if not lengths:
            raise ValueError("documents、embeddings、metadatas 至少需要传入一个")
        if len(set(lengths)) > 1:
            raise ValueError("documents、embeddings、metadatas、ids 同时传入时长度必须一致")

        count = lengths[0]
        if ids is None:
            ids = [str(uuid4()) for _ in range(count)]

        data = {"ids": ids}
        if documents is not None:
            data["documents"] = documents
        if embeddings is not None:
            data["embeddings"] = embeddings
        if metadatas is not None:
            data["metadatas"] = metadatas

        batch_size = await asyncio.to_thread(self.client.get_max_batch_size)
        for start in range(0, count, batch_size):
            batch = {key: values[start:start + batch_size] for key, values in data.items()}
            await asyncio.to_thread(self.collection.add, **batch)
        return ids

    async def get(self, **filters) -> ChromaGetResponse:
        """
        查询数据（Read）。默认排除 embeddings，无筛选条件时返回全量数据。
        筛选项透传 ChromaDB，常用键：ids / where / limit / offset / where_document / include。
        offset从0开始。
        :return: 仅含 ids、documents、embeddings、metadatas 四个字段的响应
        """
        filters.setdefault("include", ["metadatas", "documents"])
        res = await asyncio.to_thread(self.collection.get, **filters)
        return ChromaGetResponse(
            ids=res["ids"],
            documents=res.get("documents"),
            embeddings=res.get("embeddings"),
            metadatas=res.get("metadatas"),
        )

    async def query(self, query_embeddings: list[list[float]], n_results: int,
                    where: dict | None = None) -> ChromaQueryResponse:
        """
        按向量相似检索（Read）。嵌入向量由外部计算后传入。
        :param where: 可选，按 metadata 筛选
        :return: ChromaQueryResponse，含 ids/documents/metadatas/distances，
                 供 rerank 与 parent_id 溯源（query 单条，取第一组结果展平）
        """
        kwargs = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        }
        if where is not None:
            kwargs["where"] = where
        res = await asyncio.to_thread(self.collection.query, **kwargs)
        return ChromaQueryResponse(
            ids=res["ids"][0],
            documents=res["documents"][0],
            metadatas=res["metadatas"][0],
            distances=res["distances"][0],
        )

    async def update(self, ids: list[str],
                     documents: list[str] | None = None,
                     embeddings: list[list[float]] | None = None,
                     metadatas: list[dict] | None = None):
        """
        更新数据（Update）。ids 必传，documents / embeddings / metadatas 至少传一个。
        """
        if documents is None and embeddings is None and metadatas is None:
            raise ValueError("documents、embeddings、metadatas 至少需要传入一个")
        data = {"ids": ids}
        if documents is not None:
            data["documents"] = documents
        if embeddings is not None:
            data["embeddings"] = embeddings
        if metadatas is not None:
            data["metadatas"] = metadatas
        await asyncio.to_thread(self.collection.update, **data)

    async def delete(self, ids: list[str]):
        """
        按 ID 删除（Delete）。
        """
        await asyncio.to_thread(self.collection.delete, ids=ids)





if __name__ == '__main__':
    async def main():
        import chromadb
        db = ChromaBase(chromadb.EphemeralClient())
        # Create
        ids = await db.add(documents=['Hello', '扣你吉瓦', 'ohhhhhhh'],
                           metadatas=[{'src': 'a'}, {'src': 'b'}, {'src': 'a'}])
        print('added:', ids)
        # Read - 全量查询
        print('get all:', await db.get())
        # Read - 按 ID 筛选
        print('get by ids:', await db.get(ids=ids[:1]))
        # Read - 按 metadata 筛选
        print('get by where:', await db.get(where={'src': 'a'}))
        # Read - 按文档内容筛选
        print('get by where_document:', await db.get(where_document={'$contains': 'Hello'}))
        # Read - limit + offset 分页
        print('get with limit/offset:', await db.get(limit=3, offset=1))
        # Update
        await db.update(ids=ids[:1], documents=['Hello Updated'])
        print('after update:', await db.get(ids=ids[:1]))
        # Delete
        await db.delete(ids[:1])
        print('after delete:', await db.get(ids=ids))
        # Read - limit + offset 分页
        print('get with limit/offset:', await db.get(limit=10, offset=0))

    asyncio.run(main())
