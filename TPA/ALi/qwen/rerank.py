# -*- coding: utf-8 -*-
"""
文件名：rerank.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/3 15:29
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""

import dashscope
from http import HTTPStatus

from app.config.security import secure

# 以下为华北2（北京）地域的配置，调用时请将{WorkspaceId}替换为真实的业务空间ID，各地域的配置不同。
dashscope.base_http_api_url = "https://llm-udh4a0m8ljmmeirn.cn-beijing.maas.aliyuncs.com/api/v1"

def text_rerank():
    resp = dashscope.TextReRank.call(
        api_key=secure.QWEN_API_KEY,
        model="qwen3.7-text-rerank",
        query="hello",
        return_documents=True,
        documents=[
            "1161145你好",
            "扣你吉瓦",
            "hello 小明"
        ],
        top_n=10,
        instruct="Given a web search query, retrieve relevant passages that answer the query."
    )
    if resp.status_code == HTTPStatus.OK:
        print(resp)
    else:
        print(resp)

if __name__ == '__main__':
    text_rerank()
"""
{
    "status_code": 200,
    "request_id": "4b0805c0-6b36-490d-8bc1-4365f4c89905",
    "code": "",
    "message": "",
    "output": {
        "results": [
            {
                "index": 0,
                # 得到的是相关性得分，越高越相关   
                "relevance_score": 0.9334521178273196
            },
            {
                "index": 2,
                "relevance_score": 0.34100082626411193
            }
        ]
    },
    "usage": {
        "prompt_tokens": 79,
        "total_tokens": 79
    }
}
    
"""