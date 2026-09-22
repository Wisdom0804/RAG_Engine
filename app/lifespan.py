"""应用与 CLI 共用的显式资源装配、原子挂载与卸载。"""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from app.service.rag_service.service import RAGService
from tool.rag_engine.model_base import model_hub

# ponytail: 进程级独占生命周期；需要并行 APP 时改为各 APP 独立资源容器。
_lifecycle_lock = Lock()


def _load_models(config) -> tuple[object | None, object | None]:
    embedder = reranker = None
    if config.EMBEDDING_MODE == 'local':
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(config.LOCAL_EMBEDDING_NAME)
    if config.RERANK_MODE == 'local':
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(config.LOCAL_RERANK_NAME)
    return embedder, reranker


def _load_resources(config):
    import chromadb
    from root import ROOT_DIR
    from tool.rag_engine.parsing_base import DocumentParser
    from tool.siem_tool.snow_flake import SnowFlake

    embedder, reranker = _load_models(config)
    client = chromadb.PersistentClient(path=str(ROOT_DIR / 'docs' / 'vector' / 'chromadb'))
    client.get_or_create_collection('learning')
    return embedder, reranker, client, DocumentParser(), SnowFlake(config.MACHINE_ID)


@asynccontextmanager
async def lifespan(app: FastAPI | None, *, config=None) -> AsyncIterator[RAGService]:
    if not _lifecycle_lock.acquire(blocking=False):
        raise RuntimeError('模型生命周期已活跃，不支持同进程并行 APP')
    if model_hub.ready:
        _lifecycle_lock.release()
        raise RuntimeError('ModelHub 已挂载，不能重复启动生命周期')
    loading = database = service = None
    embedder = reranker = client = parser = generator = None
    try:
        if config is None:
            from app.config.security import Secure
            config = Secure()
        config.validate_models()
        loading = asyncio.create_task(asyncio.to_thread(_load_resources, config))
        embedder, reranker, client, parser, generator = await asyncio.shield(loading)
        loading = None
        from app.model import Database
        from app.model.parent_chunk import ParentChunkStore
        from tool.rag_engine.chroma_base import ChromaBase
        from tool.rag_engine.rag_base import RAGBase

        database = Database(config.POSTGRE_DATABASE_URL) if config.POSTGRE_DATABASE_URL else None
        parents = ParentChunkStore(database) if database is not None else None
        model_hub.mount(config, embedder=embedder, reranker=reranker)
        embedder = reranker = None

        def create_rag(collection_name: str, *, create: bool) -> RAGBase:
            return RAGBase(chroma=ChromaBase(client, collection_name, create=create),
                           parser=parser, model=model_hub, next_id=generator.next_id, parents=parents)

        service = RAGService(create_rag)
        if app is not None:
            app.state.rag_service = service
        yield service
    finally:
        # to_thread 无法停止构造；取消时等它结束，再释放独占生命周期。
        if loading is not None:
            while not loading.done():
                try:
                    await asyncio.shield(loading)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not loading.cancelled():
                loading.exception()
        loading = None
        if service is not None:
            service.rag_factory = None
        if app is not None:
            app.state.rag_service = None
        try:
            if database is not None:
                await database.engine.dispose()
        finally:
            model_hub.unload()
            embedder = reranker = client = parser = generator = None
            _lifecycle_lock.release()
