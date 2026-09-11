# -*- coding: utf-8 -*-
"""
文件名：security.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/3 15:05
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from root import ROOT_DIR


class Secure(BaseSettings):
    # 环境(CONSOLE:正式，DEV：测试)
    ENVIRONMENT: str
    # 服务配置信息
    APP_PORT: int
    APP_HOST: str
    APP_DEBUG: bool
    MACHINE_ID: int

    # PostgreSQL连接配置
    POSTGRE_DATABASE_URL: str
    POSTGRE_HOST: str
    POSTGRE_PORT: str
    POSTGRE_USER: str
    POSTGRE_PASSWORD: str
    POSTGRE_DB: str

    # 阿里模型
    QWEN_API_KEY: str = ''
    QWEN_API_BASE: str = ''
    QWEN_EMBEDDING_NAME: str = ''
    QWEN_RERANK_NAME: str = ''

    # 本地模型
    LOCAL_EMBEDDING_NAME: str = ''
    LOCAL_RERANK_NAME: str = ''
    EMBEDDING_MODE: Literal['local', 'api'] = 'local'
    RERANK_MODE: Literal['local', 'api'] = 'local'

    @model_validator(mode='after')
    def validate_models(self) -> Self:
        """仅校验启用模式需要的配置，不在错误信息中暴露配置值。"""
        for kind in ('EMBEDDING', 'RERANK'):
            mode = getattr(self, f'{kind}_MODE')
            required = ([f'LOCAL_{kind}_NAME'] if mode == 'local' else
                        [f'QWEN_{kind}_NAME', 'QWEN_API_KEY', 'QWEN_API_BASE'])
            for name in required:
                if not getattr(self, name).strip():
                    raise ValueError(f'{kind}_MODE={mode} 要求 {name} 非空')
        return self

    # 使用 Pydantic v2 的配置方式
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),  # 读取 .env 文件
        env_file_encoding="utf-8",
        extra="ignore"  # 避免多余字段报错
    )


secure = Secure()
