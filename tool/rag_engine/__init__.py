# -*- coding: utf-8 -*-
"""
文件名：__init__.py.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/5 10:03
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""

import chromadb
from root import ROOT_DIR

# 模块导入实现单例模式；RAG 引擎需落盘持久化，故使用 PersistentClient
# chromadb_client = chromadb.PersistentClient(path=str(ROOT_DIR / 'docs' / 'vector' / 'chromadb'))
chromadb_client = chromadb.EphemeralClient()  # 内存(重启即丢，仅用于临时调试)