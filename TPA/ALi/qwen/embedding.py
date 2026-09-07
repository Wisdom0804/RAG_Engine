# -*- coding: utf-8 -*-
"""
文件名：embedding.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/3 15:20
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
import dashscope
from http import HTTPStatus

from app.config.security import secure
from tool.rag_engine.chroma_base import ChromaBase

# 以下为华北2（北京）地域的配置，调用时请将{WorkspaceId}替换为真实的业务空间ID，各地域的配置不同。
dashscope.base_http_api_url = "https://llm-udh4a0m8ljmmeirn.cn-beijing.maas.aliyuncs.com/api/v1"

resp = dashscope.TextEmbedding.call(
    api_key=secure.QWEN_API_KEY,
    model="qwen3.7-text-embedding-flash",
    input='hello world',  # 可使用字符串列表，最长20
    dimension=1024,  # 指定向量维度（仅 qwen3.7-text-embedding、text-embedding-v3及 text-embedding-v4支持该参数）
    test_type="document",  # 检索使用query，入库、聚类、分类使用document
    output_type="dense&sparse"
)
res_hello = resp["output"]["embeddings"][0]["embedding"]
resp = dashscope.TextEmbedding.call(
    api_key=secure.QWEN_API_KEY,
    model="qwen3.7-text-embedding-flash",
    input='24688',  # 可使用字符串列表，最长20
    dimension=1024,  # 指定向量维度（仅 qwen3.7-text-embedding、text-embedding-v3及 text-embedding-v4支持该参数）
    test_type="document",  # 检索使用query，入库、聚类、分类使用document
    output_type="dense&sparse"
)
res_24688 = resp["output"]["embeddings"][0]["embedding"]
def test():
    """
    测试数据
    """
    db = ChromaBase()
    db.collection.add(
        ids=['id1', 'id2'],
        # 0.79   1.52  默认模型，嵌入方式
        # 0.85   1.48  千问模型，嵌入方式
        # 0.92   1.58  千问模型，注入向量
        embeddings=[res_24688, res_hello],
        documents=['24688','hello world']
    )
    res = db.collection.query(
        query_texts='hi',
        n_results=2
    )
    print(res)

if __name__ == '__main__':
    test()

# print(resp) if resp.status_code == HTTPStatus.OK else print(resp)

"""
{
    "status_code": 200,
    "request_id": "1ba94ac8-e058-99bc-9cc1-7fdb37940a46",
    "code": "",
    "message": "",
    "output": {
        "embeddings": [
            {
                "embedding": [-0.006929283495992422, -0.005336422007530928, ...],
                "text_index": 0
            }
        ]
    },
    "usage": {
        "prompt_tokens": 27,
        "total_tokens": 27
    }
}

"""