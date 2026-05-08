# InfiniBench 仪表盘（dashboard-ui）

## 启动

```bash
cd dashboard-ui
npm install
npm run dev
```

浏览器打开终端里提示的本地地址（一般为 `http://localhost:5173`）。

若更新了仓库根目录下的评测 CSV，可先生成前端数据再启动：

```bash
npm run generate:data
npm run dev
```

生产构建：`npm run build`（会先执行 `generate:data`）。
