"""统一响应及全局异常处理。"""
import logging
from http import HTTPStatus
from typing import Generic, Self, TypeVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.libs.enums import ResponseCode

T = TypeVar("T")
logger = logging.getLogger(__name__)


class BaseResponse(BaseModel, Generic[T]):
    code: ResponseCode
    message: str
    data: T | None = None

    @classmethod
    def success(cls, data: T | None = None, message: str = "success") -> Self:
        return cls(code=ResponseCode.SUCCESS, message=message, data=data)

    @classmethod
    def error(
        cls,
        code: ResponseCode = ResponseCode.INTERNAL_ERROR,
        message: str = "服务器内部错误",
    ) -> Self:
        code = ResponseCode(code)
        if code == ResponseCode.SUCCESS:
            raise ValueError("错误响应不能使用成功码")
        return cls(code=code, message=message, data=None)


class AppException(HTTPException):
    def __init__(
        self,
        message: str = "失败",
        code: ResponseCode = ResponseCode.BUSINESS_ERROR,
        status_code: int = 400,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.code = ResponseCode(code)
        if self.code == ResponseCode.SUCCESS:
            raise ValueError("业务异常不能使用成功码")
        super().__init__(status_code=status_code, detail=message, headers=headers)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    body = BaseResponse[None].error(code=exc.code, message=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"),
                        headers=exc.headers)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = BaseResponse[None].error(code=ResponseCode.VALIDATION_ERROR, message="请求参数错误")
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    try:
        message = HTTPStatus(exc.status_code).phrase
    except ValueError:
        message = "HTTP 请求错误"
    body = BaseResponse[None].error(code=ResponseCode.HTTP_ERROR, message=message)
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"),
                        headers=exc.headers)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("未处理的服务端异常", exc_info=(type(exc), exc, exc.__traceback__))
    body = BaseResponse[None].error()
    return JSONResponse(status_code=500, content=body.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
