# -*- coding: utf-8 -*-
"""
文件名：run.py.py
文件描述: 项目启动入口
作者: 郑智文
创建日期: 2026/9/10 17:59
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""

if __name__ == "__main__":
    import uvicorn

    from app.config.security import secure

    uvicorn.run(
        "app:create_app",
        host=secure.APP_HOST,
        port=secure.APP_PORT,
        reload=secure.APP_DEBUG,
        workers=1,
        factory=True,
    )
