# -*- coding: utf-8 -*-
"""
文件名：parsing_base.py
文件描述: 文档解析基类，包装 MarkItDown 将各类文件统一转换为 markdown 文本
作者: 郑智文
创建日期: 2026/9/7 14:22
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
import asyncio
from pathlib import Path

from markitdown import MarkItDown

from root import ROOT_DIR


class DocumentParser:
    """
    文档解析器：将各类文件统一转换为 markdown 文本。
    包装 MarkItDown 单例，提供 async 接口（阻塞 convert 经 to_thread 丢线程池）。
    """

    # 支持的扩展名白名单
    SUPPORTED_EXT = {
        '.md', '.markdown', '.txt', '.html', '.htm',
        '.csv', '.json', '.xml',
        '.docx', '.pdf', '.xlsx',
    }

    def __init__(self):
        # 零配置：使用 markitdown 内置本地转换器，不启用插件/LLM
        self._md = MarkItDown()

    async def parse(self, path: str | Path) -> str:
        """
        解析文件，返回 markdown 文本。
        :param path: 文件路径
        :return: markdown 格式文本
        :raises ValueError: 不支持的文件扩展名
        """
        ext = Path(path).suffix.lower()
        if ext not in self.SUPPORTED_EXT:
            raise ValueError(f"不支持的文件类型: {ext}")
        result = await asyncio.to_thread(self._md.convert, str(path))
        return result.markdown


# 模块级单例
document_parser = DocumentParser()


if __name__ == '__main__':
    async def main():
        # 演示：解析本文件自身（.py 不在白名单，演示白名单拦截）
        try:
            await document_parser.parse(__file__)
        except ValueError as e:
            print(f"白名单拦截: {e}")

        design_doc = ROOT_DIR / 'docs' / 'file' / '郑智文.docx'
        if design_doc.exists():
            markdown = await document_parser.parse(design_doc)
            print(f"解析成功，markdown 长度: {len(markdown)} 字符")
            print(f"\n{markdown}")

    asyncio.run(main())
