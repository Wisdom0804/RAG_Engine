"""测试接口：接收请求，将 Service 返回的 VO 包装为统一响应。"""
from fastapi import APIRouter

from app.Schemas.test import TestDTO, TestVO
from app.libs.error_code import Success
from app.service.test_service import TestService

router = APIRouter()
service = TestService()


@router.post("/test", response_model=Success[TestVO])
async def test(data: TestDTO):
    """将校验后的请求传给测试服务，并返回包含测试结果的成功响应。"""
    result = await service.process(data)
    return Success(data=result)
