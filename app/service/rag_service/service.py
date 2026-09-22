"""RAG 文档入库和上下文检索服务。"""
import asyncio
import hashlib
import tempfile
from collections.abc import Callable, Coroutine
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import UploadFile

from app.Schemas.rag import IngestDTO, IngestVO, RetrieveDTO, RetrieveVO
from app.libs.error_code import Error

if TYPE_CHECKING:
    from tool.rag_engine.rag_base import RAGBase

MAX_FILE_SIZE = 20 * 1024 * 1024
_active_collections: set[str] = set()
_active_lock = Lock()
T = TypeVar("T")


async def _finish_before_cancel(operation: Coroutine[Any, Any, T]) -> T:
    """取消请求时等待正在使用临时文件或写库的工作结束，再向外传播取消。"""
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


class RAGService:
    """校验请求并复用 RAG 引擎，模型由应用生命周期管理。"""

    def __init__(self, rag_factory: Callable[..., "RAGBase"] | None = None):
        self.rag_factory = rag_factory

    async def _get_rag(self, collection_name: str, *, create: bool) -> "RAGBase":
        """在线程中使用装配点提供的工厂打开集合。"""
        from chromadb.errors import NotFoundError

        if self.rag_factory is None:
            raise Error('服务尚未就绪', status_code=503)
        try:
            return await asyncio.to_thread(self.rag_factory, collection_name, create=create)
        except NotFoundError as exc:
            raise Error('集合不存在', status_code=404) from exc

    async def ingest(self, file: UploadFile, params: IngestDTO) -> IngestVO:
        """校验文件后同步入库；同集合内按文件内容 SHA-256 拒绝重复。

        Args:
            file: 单个上传文件，最大 20 MiB。
            params: 集合名与切分参数。

        Returns:
            集合、文件名与已写入的块数。

        Raises:
            Error: 参数无效、文件重复、集合忙或策略不一致。
        """
        from tool.rag_engine.parsing_base import DocumentParser

        file_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(file_name).suffix.lower()
        if suffix not in DocumentParser.SUPPORTED_EXT:
            raise Error(f"不支持的文件类型: {suffix}", status_code=415)
        if file.size is not None and file.size > MAX_FILE_SIZE:
            raise Error("文件大小不能超过 20 MiB", status_code=413)
        data = await file.read(MAX_FILE_SIZE + 1)
        if len(data) > MAX_FILE_SIZE:
            raise Error("文件大小不能超过 20 MiB", status_code=413)
        if not data:
            raise Error("文件内容不能为空", status_code=422)
        # 单进程部署下，同集合写入互斥；不同集合可同时入库。
        with _active_lock:
            if params.collection_name in _active_collections:
                raise Error("集合正在入库，请稍后重试", status_code=409)
            _active_collections.add(params.collection_name)
        try:
            return await _finish_before_cancel(self._ingest(data, file_name, suffix, params))
        finally:
            with _active_lock:
                _active_collections.remove(params.collection_name)

    async def _ingest(self, data: bytes, file_name: str, suffix: str, params: IngestDTO) -> IngestVO:
        """在集合写入互斥期间完成持久化约束检查和文件处理。"""
        rag = await self._get_rag(params.collection_name, create=True)
        from tool.rag_engine.chroma_base import CollectionConflict
        from tool.rag_engine.model_base import ModelNotReady
        from tool.rag_engine.rag_base import StrategyUnavailable

        digest = await asyncio.to_thread(lambda: hashlib.sha256(data).hexdigest())
        kwargs = params.model_dump(include={'chunk_size', 'overlap', 'threshold', 'parent_chunk_size'},
                                   exclude_none=True)
        with tempfile.TemporaryDirectory(prefix="rag-upload-") as folder:
            path = Path(folder) / f"document{suffix}"
            await asyncio.to_thread(path.write_bytes, data)
            try:
                count = await rag.ingest(str(path), params.strategy,
                                         source_file=file_name, file_sha256=digest, **kwargs)
            except CollectionConflict as exc:
                raise Error(str(exc), status_code=409) from exc
            except ModelNotReady as exc:
                raise Error('模型尚未就绪', status_code=503) from exc
            except StrategyUnavailable as exc:
                raise Error(str(exc), status_code=422) from exc
            except ValueError as exc:
                raise Error("文档解析或切分失败，请检查文件及参数", status_code=422) from exc
        return IngestVO(collection_name=params.collection_name, file_name=file_name, chunk_count=count)

    async def retrieve(self, data: RetrieveDTO) -> RetrieveVO:
        """检查集合与模型状态并返回有序上下文原文。

        Args:
            data: 查询、集合、候选数、重排开关与过滤条件。

        Returns:
            上下文原文与实际数量；无命中时返回空列表。

        Raises:
            Error: 参数无效、集合不存在或模型尚未就绪。
        """
        rag = await self._get_rag(data.collection_name, create=False)
        from tool.rag_engine.model_base import ModelNotReady
        from tool.rag_engine.rag_base import StrategyUnavailable

        try:
            contexts = await rag.retrieve(data.query, data.top_k, data.rerank, data.where)
        except ModelNotReady as exc:
            raise Error('模型尚未就绪', status_code=503) from exc
        except StrategyUnavailable as exc:
            raise Error(str(exc), status_code=422) from exc
        return RetrieveVO(collection_name=data.collection_name, query=data.query,
                          count=len(contexts), contexts=contexts)
