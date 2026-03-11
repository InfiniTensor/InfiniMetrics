# InfiniMetrics Dashboard 使用指南

## 1. Dashboard 简介

InfiniMetrics Dashboard 用于统一展示 AI 加速卡在以下场景下的测试与评测结果

- 通信（NCCL / 集合通信）
- 训练（Training / 分布式训练）
- 推理（Direct / Service 推理）
- 算子（核心算子性能）

测试框架输出两类数据：
```
JSON  -> 配置 / 环境 / 标量指标
CSV   -> 曲线 / 时序数据
```
Dashboard 会自动加载测试结果，并提供统一的分析功能，包括：

- Run ID 模糊搜索：支持通过部分 Run ID 快速定位测试运行

- 通用筛选器：按框架、模型、设备数量等条件筛选

- 多运行对比：同时选择多个测试运行进行性能对比

- 性能可视化：展示 latency / throughput / loss 等性能曲线

- 统计与配置展示：查看吞吐量统计、运行配置和环境信息

例如可以输入：
```
allreduce
service
```
对 Run ID 进行模糊匹配搜索

示例截图：
![run_id搜索](https://private-user-images.githubusercontent.com/94775646/561443423-ac18e86d-b6e3-4ae5-a386-15d9cf34bf69.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0MjMtYWMxOGU4NmQtYjZlMy00YWU1LWEzODYtMTVkOWNmMzRiZjY5LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI3NDFjM2QzZThjMGY3MjM3ZTJkY2IxMDY4ZDc2ZWYxZmYzOGQzZDAyYjQyZDE4NGZmZmEwZDRjOTQzN2IzOGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.pOFgY_W8dwv-cMMSBP4yee_2l0we_iCL4jptlMLIsnk)
## 2. 运行 Dashboard 
### 2.1 环境依赖
使用 Dashboard 前需要安装以下依赖：
```
streamlit
plotly
pandas
```
### 2.2 启动 Dashboard
在项目根目录执行：
```
python -m streamlit run dashboard/app.py
```
访问地址，启动成功后显示：
```
Local URL:    http://localhost:8501
Network URL:  http://<server-ip>:8501
```
说明：

Local URL：仅本机访问

Network URL：同一网络内其他机器可访问

## 3. 通信测试分析
路径：

```
Dashboard → 通信性能测试
```

支持：
```
带宽分析曲线 - 峰值带宽

延迟分析曲线 - 平均延迟

测试耗时

显存使用

通信配置解析
```

示例截图：

![通信测试](https://private-user-images.githubusercontent.com/94775646/561443439-b3e3f409-6bb9-4c32-bee8-9928781fcd39.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0MzktYjNlM2Y0MDktNmJiOS00YzMyLWJlZTgtOTkyODc4MWZjZDM5LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM1MGU2YTcxNjcxNWM4ZWIwYWJiZTVkNmU2NTIxNWNkZDI3NGRiNTNmNWI0YWY4NWUxZmI0MzBjZjc4OGVkZjgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.AaFHdRt4SGyldFCZR1hNBhySRoR2cGDHQ0aAVNG4dq8)
## 4. 推理测试分析

路径：

```
Dashboard → 推理性能测试
```

模式：
```
Direct Inference
Service Inference
```
展示指标：
```
TTFT

Latency

Throughput

显存使用

推理配置解析
```
示例截图：

![推理测试](https://private-user-images.githubusercontent.com/94775646/561443450-9b57045c-d320-4da6-9b23-6e0f9a6c96cb.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0NTAtOWI1NzA0NWMtZDMyMC00ZGE2LTliMjMtNmUwZjlhNmM5NmNiLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcxYTVjOWY2MzE4NjNjYjJlNGI3ZjkzMDk4MzJmZWNkY2ZiZWI2ZjRiZjcyZjVkZTAzNGE0YzljYjVhZmJhZDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0._LAZMfKkStSoWgw-1t22fOdlbkng-xZnV9rHIpd4v-M)

## 5. 训练测试分析
路径：

```
Dashboard → 训练性能测试
```

支持：
```
Loss 曲线

Perplexity 曲线

Throughput 曲线

显存使用

训练配置解析
```
示例截图：

![训练测试](https://private-user-images.githubusercontent.com/94775646/561443390-5a789775-9fd4-4146-932d-df2436c20977.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDMzOTAtNWE3ODk3NzUtOWZkNC00MTQ2LTkzMmQtZGYyNDM2YzIwOTc3LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc4ZjFkZGFjYWE1ZWM0ZDRjOWQ3M2QxYzM4YzRjNzUyZmMzZGIwMTQwNjBjMDU4N2QyMmUyOTRlYzU3MWRkN2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.3YfafricSVbHzguSeDhhKiQ0S68AQDY48a1HRpMHTj4)

## 6. 算子测试分析

路径：

```
Dashboard → 算子性能测试
```

支持：
```
latency

flops

bandwidth
```

示例截图：

![算子测试](https://private-user-images.githubusercontent.com/94775646/561443431-58d7f944-8e6b-4f2a-8fdc-671d09653da8.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0MzEtNThkN2Y5NDQtOGU2Yi00ZjJhLThmZGMtNjcxZDA5NjUzZGE4LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBjNzNhNGI5YjhlMDc3MjNiMzU3ODQ5YTNjMDBjYTcyN2E1NjIxNjAyNGU3ZDM5ODMwMGVlZTMyMWY0OGE3ZTcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OpDxdAaEsdHYX_jOWNmmdPB-jKp92gbLh1VFs0H2j6c)

