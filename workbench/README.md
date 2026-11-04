# SpineVision Workbench

SpineVision Agent 的 React/Vinext 操作工作台，用于病例上传、MRI/Mask 阅片、Agent 轨迹查看、RAG 证据核对和报告人工复核。

## 启动

```bash
cp .env.example .env.local
pnpm install
pnpm dev
```

默认连接 `http://127.0.0.1:8000`。可通过 `NEXT_PUBLIC_SPINEVISION_API_URL` 配置后端地址，也可在页面设置面板中临时修改。

## 检查

```bash
pnpm lint
pnpm test
```

工作台只展示研究型模型输出；低置信度、证据缺失或 Verifier 冲突必须进入人工复核。
