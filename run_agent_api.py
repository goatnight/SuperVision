"""本地启动 FastAPI；生产环境应使用受控进程管理和持久化状态库。"""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("spinevision_agent.api:app", host="127.0.0.1", port=8000, reload=False)
