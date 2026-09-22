"""RAG HTTP 契约：校验请求，将 Service 返回的 VO 包装为成功响应。"""
from typing import Annotated

from fastapi import APIRouter, Form, Request

from app.Schemas.rag import IngestVO, RetrieveDTO, RetrieveVO, UploadDocumentDTO
from app.libs.error_code import Success
from app.libs.error_code import Error

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/documents", response_model=Success[IngestVO])
async def ingest_document(
    request: Request,
    data: Annotated[UploadDocumentDTO, Form(media_type="multipart/form-data")],
) -> Success[IngestVO]:
    """上传文件并建立 RAG 向量索引。

    Args:
        data: 已由 Pydantic 校验的上传文件、集合名和切分配置。

    Returns:
        包含入库结果 VO 的统一成功响应。

    """
    service = getattr(request.app.state, 'rag_service', None)
    if service is None:
        raise Error('服务尚未就绪', status_code=503)
    return Success(data=await service.ingest(data.file, data))


@router.post("/retrieve", response_model=Success[RetrieveVO])
async def retrieve_context(request: Request, data: RetrieveDTO) -> Success[RetrieveVO]:
    """检索相关上下文原文，不生成答案。

    Args:
        data: 检索请求 DTO。

    Returns:
        包含有序上下文与实际数量的统一成功响应。
    """
    service = getattr(request.app.state, 'rag_service', None)
    if service is None:
        raise Error('服务尚未就绪', status_code=503)
    return Success(data=await service.retrieve(data))
