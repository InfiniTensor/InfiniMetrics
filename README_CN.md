# InfiniMetrics

<div align="center">

**面向 InfiniTensor 的全面加速器评估框架**

[![Format Check](https://img.shields.io/badge/Format_Check-passing-success)](https://github.com/InfiniTensor/InfiniMetrics)
[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniMetrics)](https://github.com/InfiniTensor/InfiniMetrics/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniMetrics)](https://github.com/InfiniTensor/InfiniMetrics/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/InfiniTensor/InfiniMetrics/blob/master/LICENSE)

一个统一、模块化的测试框架，专为加速卡和软件栈的全面性能评估而设计。

</div>

---

## 🎯 项目概述

**InfiniMetrics** 提供了标准化的接口，用于在多个层次进行基准测试：

- **硬件层**：GPU 内存带宽、缓存性能、计算能力
- **算子层**：单个操作的性能（FLOPS、延迟）
- **推理层**：端到端模型推理吞吐量和延迟
- **通信层**：NCCL 集合操作和 GPU 间通信

### 核心特性

- **统一适配器接口** - 所有测试类型和框架的一致 API
- **可扩展架构** - 易于添加新的测试类型、框架和指标
- **全面的指标系统** - 标量值、时间序列数据、自定义测量
- **框架无关** - 支持 InfiniLM、vLLM、InfiniCore 等
- **生产就绪** - 健壮的错误处理、日志记录和结果聚合

---

## 📚 文档

详细的指南、配置说明和示例，请参阅[完整文档](./docs/zh)。

### 快速链接

- [安装指南](./docs/zh/installation.md) - 前置要求和依赖
- [配置指南](./docs/zh/configuration.md) - 输入格式和参数

---

## 🤝 贡献

欢迎贡献！请参阅我们的[贡献指南](./docs/zh/development.md)了解详情。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](./LICENSE) 文件。

---

<div align="center">

**由 InfiniTensor 团队构建 ❤️**

[官网](https://infinitensor.org) | [文档](./docs/zh) | [GitHub](https://github.com/InfiniTensor)

</div>
