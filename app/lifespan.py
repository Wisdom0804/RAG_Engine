"""应用与 CLI 共用的模型生命周期：加载、原子挂载、卸载。"""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config.security import secure
from tool.rag_engine.model_base import model_hub
from tool.rag_engine.splitter_base import text_splitter


# ponytail: 进程级独占生命周期；需要并行 APP 时改为各 APP 独立资源容器。
_lifecycle_lock = Lock()


def _load_models() -> tuple[object | None, object | None]:
    embedder = reranker = None
    try:
        if secure.EMBEDDING_MODE == 'local':
            embedder = SentenceTransformer(secure.LOCAL_EMBEDDING_NAME)
        if secure.RERANK_MODE == 'local':
            reranker = CrossEncoder(secure.LOCAL_RERANK_NAME)
        return embedder, reranker
    finally:
        # 加载第二个模型失败时，也释放第一个模型的局部引用。
        embedder = reranker = None


@asynccontextmanager
async def lifespan(app: FastAPI | None) -> AsyncIterator[None]:
    if not _lifecycle_lock.acquire(blocking=False):
        raise RuntimeError('模型生命周期已活跃，不支持同进程并行 APP')
    if model_hub.ready:
        _lifecycle_lock.release()
        raise RuntimeError('ModelHub 已挂载，不能重复启动生命周期')
    loading = None
    embedder = reranker = None
    try:
        secure.validate_models()
        loading = asyncio.create_task(asyncio.to_thread(_load_models))
        embedder, reranker = await asyncio.shield(loading)
        model_hub.mount(embedder=embedder, reranker=reranker)
        loading = None
        embedder = reranker = None
        yield
    finally:
        # to_thread 无法停止正在构造的模型；取消时等待它结束再允许下一次启动。
        if loading is not None:
            while not loading.done():
                try:
                    await asyncio.shield(loading)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not loading.cancelled():
                loading.exception()  # 消费取消期间发生的加载异常。
        loading = None
        embedder = reranker = None
        try:
            text_splitter.clear_cache()
        finally:
            model_hub.unload()
            _lifecycle_lock.release()
