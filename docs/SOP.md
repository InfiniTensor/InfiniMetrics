# InfiniMetrics 启动 SOP

本文档描述如何启动 InfiniMetrics 的各个组件。

## 架构概览

```
┌─────────────────┐     ┌─────────────────┐
│  JSON 文件      │     │   Dashboard     │
│  (output/)      │     │   (Streamlit)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  FileWatcher    │     │  DataLoader     │
│  (数据监听)     │     │  (数据加载)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│              db 模块                     │
└────────────────┬────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │   MongoDB     │
         └───────────────┘
```

## 1. 前置条件

### 1.1 安装依赖

```bash
pip install pymongo streamlit pandas watchdog plotly
```

### 1.2 确保 MongoDB 运行

**方式 A：Docker（推荐开发环境）**

```bash
# 启动 MongoDB 容器（带数据持久化）
docker run -d \
  -p 27017:27017 \
  -v ~/mongodb_data:/data/db \
  --name mongodb \
  mongo:latest

# 查看状态
docker ps | grep mongodb

# 停止
docker stop mongodb

# 启动
docker start mongodb
```

**方式 B：直接安装（推荐生产环境）**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb

# 验证
mongod --version
```

**方式 C：Windows**

```bash
# 下载 MongoDB Community Server
# https://www.mongodb.com/try/download/community

# 或使用 Chocolatey
choco install mongodb

# 启动服务
net start MongoDB
```

### 1.3 环境变量配置（可选）

```bash
# 默认值已内置，通常不需要配置
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="infinimetrics"
export MONGODB_COLLECTION="test_runs"
```

## 2. 启动步骤

### 2.1 导入/监听数据

**方式 A：一次性导入现有数据**

```bash
cd /path/to/InfiniMetrics

# 导入 output 和 summary_output 目录的数据
python -m db.cli import \
  --output-dir ./output \
  --summary-dir ./summary_output
```

**方式 B：启动持续监听（推荐）**

```bash
cd /path/to/InfiniMetrics

# 后台运行
nohup python -m db.cli watch start \
  --output-dir ./output \
  --summary-dir ./summary_output \
  > logs/watcher.log 2>&1 &

# 查看日志
tail -f logs/watcher.log

# 查看进程
ps aux | grep "db.cli watch"
```

**方式 C：手动触发扫描**

```bash
python -m db.cli watch scan \
  --output-dir ./output \
  --summary-dir ./summary_output
```

### 2.2 启动 Dashboard

```bash
cd /path/to/InfiniMetrics/dashboard

# 前台运行（开发调试）
streamlit run app.py --server.port 8501

# 后台运行（生产部署）
nohup streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  > ../logs/dashboard.log 2>&1 &

# 查看日志
tail -f ../logs/dashboard.log
```

访问地址：http://localhost:8501

## 3. 常用命令

### 3.1 db.cli 命令

```bash
# 查看帮助
python -m db.cli --help

# 列出所有测试记录
python -m db.cli list

# 按类型筛选
python -m db.cli list --test-type infer

# 查看详情
python -m db.cli info <run_id>

# 删除记录
python -m db.cli delete <run_id>

# 强制删除（无需确认）
python -m db.cli delete <run_id> --force
```

### 3.2 服务管理

```bash
# 查看运行中的服务
ps aux | grep -E "(streamlit|db.cli)"

# 停止 watcher
pkill -f "db.cli watch"

# 停止 dashboard
pkill -f "streamlit run"

# 停止 MongoDB (Docker)
docker stop mongodb
```

## 4. 完整启动流程（一键脚本）

创建日志目录：

```bash
mkdir -p logs
```

启动所有服务：

```bash
#!/bin/bash
# start_all.sh

PROJECT_ROOT="/path/to/InfiniMetrics"
cd "$PROJECT_ROOT"

# 1. 确保 MongoDB 运行
echo "检查 MongoDB..."
if ! docker ps | grep -q mongodb; then
    echo "启动 MongoDB..."
    docker start mongodb || docker run -d -p 27017:27017 -v ~/mongodb_data:/data/db --name mongodb mongo:latest
fi

# 2. 启动数据监听
echo "启动数据监听..."
mkdir -p logs
nohup python -m db.cli watch start \
    --output-dir ./output \
    --summary-dir ./summary_output \
    > logs/watcher.log 2>&1 &

# 3. 启动 Dashboard
echo "启动 Dashboard..."
cd dashboard
nohup streamlit run app.py --server.port 8501 \
    > ../logs/dashboard.log 2>&1 &

echo "启动完成！"
echo "Dashboard: http://localhost:8501"
echo "日志目录: $PROJECT_ROOT/logs/"
```

停止所有服务：

```bash
#!/bin/bash
# stop_all.sh

echo "停止服务..."
pkill -f "db.cli watch"
pkill -f "streamlit run"
docker stop mongodb

echo "所有服务已停止"
```

## 5. 故障排查

### 5.1 Dashboard 显示"无法连接到 MongoDB"

```bash
# 检查 MongoDB 是否运行
docker ps | grep mongodb
# 或
sudo systemctl status mongodb

# 检查端口
netstat -an | grep 27017

# 测试连接
python -c "from pymongo import MongoClient; c=MongoClient('mongodb://localhost:27017'); print(c.server_info())"
```

### 5.2 数据没有导入

```bash
# 检查 watcher 日志
tail -f logs/watcher.log

# 手动触发导入
python -m db.cli import --output-dir ./output --summary-dir ./summary_output

# 查看已导入数据
python -m db.cli list
```

### 5.3 Dashboard 页面空白

```bash
# 检查 dashboard 日志
tail -f logs/dashboard.log

# 确保有数据
python -m db.cli list

# 清除 Streamlit 缓存
streamlit cache clear
```

## 6. 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| MongoDB | 27017 | 数据库 |
| Dashboard | 8501 | Streamlit Web UI |

## 7. 目录结构

```
InfiniMetrics/
├── db/                 # MongoDB 集成模块
├── dashboard/          # Streamlit Dashboard
├── output/             # 测试结果 JSON/CSV
├── summary_output/     # Dispatcher 汇总文件
└── logs/               # 日志目录（需创建）
```
