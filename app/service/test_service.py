"""测试服务：验证 API 到 Service 的数据传递，不调用模型或数据库。"""
from app.Schemas.test import TestDTO, TestVO


class TestService:
    """处理测试接口的业务数据。"""

    async def process(self, data: TestDTO) -> TestVO:
        """将请求中的文本转为结果 VO，供 API 包装统一响应。"""
        return TestVO(text=data.text)
