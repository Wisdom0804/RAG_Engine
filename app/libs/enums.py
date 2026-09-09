# -*- coding: utf-8 -*-
"""
文件名：enums.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/9 09:07
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
from enum import Enum


class ChunkingStrategyEnum(Enum):
    """
    分块策略枚举

    系统中应该统一存放策略枚举，这里为了
    """
    定长切分 = 'fixed_length'
    语义切分 = 'semantic'
    递归切分 = 'recursive'
    结构切分 = 'structural'
    父子切分 = 'parent_child'