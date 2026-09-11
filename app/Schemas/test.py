"""测试接口的数据契约。"""
from pydantic import BaseModel, Field


class TestDTO(BaseModel):
    """接收要传给测试服务的文本。"""

    text: str = Field(min_length=1, description="用于验证接口与服务调用链的文本")


class TestVO(BaseModel):
    """测试服务返回给接口的结果。"""

    text: str = Field(description="服务处理后返回的文本")
