"""可返回的成功响应与可抛出的业务异常。"""
from typing import Generic, Literal, TypeVar

from app.libs.enums import ResponseCode
from app.libs.error import AppException, BaseResponse

T = TypeVar("T")


class Success(BaseResponse[T], Generic[T]):
    code: Literal[ResponseCode.SUCCESS] = ResponseCode.SUCCESS
    message: str = "success"


class Error(AppException):
    pass
