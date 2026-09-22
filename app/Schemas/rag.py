"""RAG 接口的请求 DTO 与响应 VO。"""
import math
from typing import Any, Literal, Self

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CollectionDTO(BaseModel):
    """使用 Chroma 集合标识的请求。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    collection_name: str = Field(
        default="learning", min_length=3, max_length=63,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*[A-Za-z0-9]$",
        description="集合名，3–63 个 ASCII 字符，首尾为字母或数字",
    )


class IngestDTO(CollectionDTO):
    """上传表单的切分配置；父子模式使用结构父块和递归子块。"""

    strategy: Literal["fixed_length", "semantic", "recursive", "structural", "parent_child"]
    chunk_size: int | None = Field(default=None, gt=0, le=100_000, description="块大小；父子模式为子块大小")
    overlap: int | None = Field(default=None, ge=0, description="相邻块重叠字符数；语义切分不支持")
    threshold: float | None = Field(default=None, ge=0, le=1, description="仅语义切分使用的相似度阈值")
    parent_chunk_size: int | None = Field(default=None, gt=0, le=100_000, description="仅父子模式使用的父块大小")

    @model_validator(mode="after")
    def validate_chunking(self) -> Self:
        """补全策略默认值，并验证跨字段约束。"""
        defaults = {"fixed_length": 200, "semantic": 1000, "recursive": 500,
                    "structural": 500, "parent_child": 200}
        self.chunk_size = self.chunk_size if self.chunk_size is not None else defaults[self.strategy]
        if self.strategy == "semantic":
            if self.overlap is not None:
                raise ValueError("semantic 不支持 overlap")
            self.threshold = self.threshold if self.threshold is not None else 0.51
        else:
            if self.threshold is not None:
                raise ValueError("threshold 仅适用于 semantic")
            default_overlap = 50 if self.strategy in {"recursive", "structural"} else 0
            self.overlap = self.overlap if self.overlap is not None else default_overlap
            if self.overlap >= self.chunk_size:
                raise ValueError("overlap 必须小于 chunk_size")
        if self.strategy == "parent_child":
            self.parent_chunk_size = self.parent_chunk_size if self.parent_chunk_size is not None else 800
            if self.chunk_size >= self.parent_chunk_size:
                raise ValueError("子块大小必须小于父块大小")
        elif self.parent_chunk_size is not None:
            raise ValueError("parent_chunk_size 仅适用于 parent_child")
        return self


class UploadDocumentDTO(IngestDTO):
    """由 FastAPI 直接解析并校验的文件上传表单。"""

    file: UploadFile = Field(description="单个文档，最大 20 MiB")


class RetrieveDTO(CollectionDTO):
    """描述一次上下文检索请求。"""

    query: str = Field(min_length=1, max_length=10_000, description="去除首尾空白后的非空查询")
    top_k: int = Field(default=5, ge=1, le=50, description="候选数；父块去重后结果可能少于此数")
    rerank: bool = Field(default=False, description="是否重排候选上下文")
    where: dict[str, Any] | None = Field(default=None, description="Chroma metadata 过滤表达式")

    @field_validator("where")
    @classmethod
    def validate_filter(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """复用存储 SDK 的过滤表达式校验，不连接向量库。"""
        if value is not None:
            from chromadb.api.types import validate_where

            validate_where(value)
            _validate_filter_values(value)
        return value


def _validate_filter_values(where: dict[str, Any]) -> None:
    """补充 SDK 未限制的操作数类型，避免合法 JSON 在存储层变成 500。"""
    def scalar(value: Any) -> bool:
        return type(value) in (str, bool, int, float) and (
            not isinstance(value, float) or math.isfinite(value))

    for key, expression in where.items():
        if key in {"$and", "$or"}:
            for child in expression:
                _validate_filter_values(child)
            continue
        if not key or key.startswith("$"):
            raise ValueError("无效的 metadata 字段")
        if not isinstance(expression, dict):
            if not scalar(expression):
                raise ValueError("过滤值必须为有限标量")
            continue
        for operator, operand in expression.items():
            if operator in {"$in", "$nin"}:
                if not all(scalar(item) and type(item) is type(operand[0]) for item in operand):
                    raise ValueError("过滤列表必须包含同类型标量")
            elif not scalar(operand):
                raise ValueError("过滤操作数必须为有限标量")
            elif operator in {"$gt", "$gte", "$lt", "$lte"} and type(operand) not in (int, float):
                raise ValueError("大小比较需要数值")


class IngestVO(BaseModel):
    """描述文档入库结果。"""

    collection_name: str = Field(description="入库集合")
    file_name: str = Field(description="原文件名，不含上传方路径")
    chunk_count: int = Field(ge=1, description="写入向量库的块数；父子模式为子块数")


class RetrieveVO(BaseModel):
    """描述检索得到的上下文原文。"""

    collection_name: str
    query: str
    count: int = Field(ge=0, description="实际返回的上下文数量")
    contexts: list[str] = Field(description="按相关性排序的原文；父子模式返回去重父块")
