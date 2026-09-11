## python版本：3.12.10

```
下包镜像网址
https://mirrors.aliyun.com/pypi/simple/
技能
npx skills@latest add mattpocock/skills
/setup-matt-pocock-skills
```

学习用项目，过程中为了测试功能，可能不会补全生产环境下需要的一些功能

## FastAPI 骨架

使用 Python **3.12.10**，在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install fastapi uvicorn httpx pydantic-settings sentence-transformers dashscope numpy chromadb
python run.py
```

已有环境可直接激活，无需重新创建。完整 RAG 演示还需要原有解析、SQLAlchemy、asyncpg 等依赖和 PostgreSQL 表；上面是启动骨架及运行测试所需依赖，测试使用模型/数据库替身。

根 `.env` 或环境变量需提供 `app/config/security.py` 的必填服务与数据库配置；`APP_HOST`、`APP_PORT` 控制监听地址，`APP_DEBUG` 仅控制 Uvicorn 热重载。FastAPI 始终关闭 debug，未知错误不会返回客户端堆栈。启动入口固定单 worker。

| 配置 | 默认值/要求 |
|---|---|
| `EMBEDDING_MODE`、`RERANK_MODE` | 各自独立选择 `local` / `api`，默认 `local` |
| `LOCAL_EMBEDDING_NAME`、`LOCAL_RERANK_NAME` | 对应 local 模式必须提供模型名称或本地路径 |
| `QWEN_EMBEDDING_NAME`、`QWEN_RERANK_NAME` | 对应 api 模式必须提供模型名，不根据前缀判断模式 |
| `QWEN_API_KEY`、`QWEN_API_BASE` | 任一 api 模式启用时必填 |

未启用模式的模型配置可为空。local 模型只在 lifespan 启动时通过工作线程加载，全部成功后挂载共享 `model_hub`；失败直接中止启动。关闭清理模型和全局切分缓存，允许重新启动；同进程不支持并行 APP 生命周期。释放引用不保证 GPU 分配器立即归还显存。

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/test -ContentType 'application/json' -Body '{"text":"hello"}'
```

`POST /test` 接收 `TestRequest` 请求体（必填非空字符串 `text`），API 将数据传给 `TestService.process()`，由 Service 返回 `TestVO`，API 使用 `Success[TestVO]` 响应模型：

```json
{"code": 0, "message": "success", "data": {"text": "hello"}}
```

业务错误/校验错误/HTTP 错误/内部错误分别使用 code 1000/1001/1002/1003；HTTP 状态保留 400/422/原状态/500，错误 data 固定为 null。文档位于 `/docs` 和 `/openapi.json`。测试接口不调用 RAG、模型或数据库，但应用仍会在接收请求前初始化所选模型。

模型业务调用方向为 API → Service → rag_base → model_hub。导入 ModelHub/RAGBase 不构造本地模型；CLI 演示显式复用 lifespan。语义切分必须注入支持同步 `encode` 的实例，API 嵌入模式下不会偷偷加载本地模型，应使用非语义策略或显式提供语义切分模型。Chroma 仍为进程内临时存储。

```powershell
python -m compileall -q app tool TPA root.py run.py
python -m unittest discover -s tests
```

第一条仅检查语法；第二条验证响应、配置、生命周期、调用链与分块行为，不下载模型、不连接数据库、不调用付费 API。真实模型启动和 RAG 演示需要另行准备模型、样本与数据库。

设计与实施记录见 [设计方案](docs/plans/2026-09-11-fastapi-skeleton.md) 和 [实施方案](docs/plans/2026-09-11-fastapi-skeleton-implementation.md)。
