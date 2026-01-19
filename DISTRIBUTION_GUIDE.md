# Tensor分布类型使用指南

## 支持的分布类型

Random Input Generator现在支持15种不同的分布类型，可以在配置文件中为每个输入指定不同的分布。

### 1. 基础分布

#### uniform (均匀分布)
```json
{
  "name": "input",
  "shape": [1024, 1024],
  "dtype": "float16",
  "_random": {
    "distribution": "uniform",
    "params": {
      "low": -1.0,    // 下限 (默认: 0.0)
      "high": 1.0,    // 上限 (默认: 1.0)
      "scale": 1.0,   // 缩放因子
      "bias": 0.0     // 偏移量
    }
  }
}
```

#### normal (正态分布)
```json
{
  "_random": {
    "distribution": "normal",
    "params": {
      "mean": 0.0,    // 均值 (默认: 0.0)
      "std": 1.0      // 标准差 (默认: 1.0)
    }
  }
}
```

#### standard_normal (标准正态分布)
```json
{
  "_random": {
    "distribution": "standard_normal",
    "params": {}
  }
}
```

#### randint (随机整数)
```json
{
  "_random": {
    "distribution": "randint",
    "params": {
      "low": -100,    // 下限 (包含)
      "high": 100     // 上限 (不包含)
    }
  }
}
```

### 2. 统计分布

#### lognormal (对数正态分布)
```json
{
  "_random": {
    "distribution": "lognormal",
    "params": {
      "mean": 0.0,    // 均值
      "std": 1.0       // 标准差
    }
  }
}
```

#### exponential (指数分布)
```json
{
  "_random": {
    "distribution": "exponential",
    "params": {
      "scale": 1.0     // 尺度参数 (默认: 1.0)
    }
  }
}
```

#### laplace (拉普拉斯分布)
```json
{
  "_random": {
    "distribution": "laplace",
    "params": {
      "loc": 0.0,      // 位置参数
      "scale": 1.0     // 尺度参数
    }
  }
}
```

#### cauchy (柯西分布)
```json
{
  "_random": {
    "distribution": "cauchy",
    "params": {
      "loc": 0.0,      // 位置参数
      "scale": 1.0     // 尺度参数
    }
  }
}
```

### 3. 离散分布

#### poisson (泊松分布)
```json
{
  "_random": {
    "distribution": "poisson",
    "params": {
      "lam": 5.0       // 期望值 (默认: 1.0)
    }
  }
}
```

#### zipf (Zipf分布)
```json
{
  "_random": {
    "distribution": "zipf",
    "params": {
      "a": 2.0         // 形状参数 (必须 > 1)
    }
  }
}
```

### 4. 特殊矩阵

#### identity (单位矩阵)
```json
{
  "_random": {
    "distribution": "identity",
    "params": {}
  }
}
```
**注意**: 只支持2D方阵（shape[0] == shape[1]）

#### orthogonal (正交矩阵)
```json
{
  "_random": {
    "distribution": "orthogonal",
    "params": {}
  }
}
```
**注意**: 只支持2D矩阵，生成随机正交矩阵

#### sparse (稀疏矩阵)
```json
{
  "_random": {
    "distribution": "sparse",
    "params": {
      "density": 0.1,  // 非零元素密度 (0-1)
      "sparsity": null  // 稀疏度 (0-1), 与density二选一
    }
  }
}
```

### 5. 常量分布

#### ones (全1)
```json
{
  "_random": {
    "distribution": "ones",
    "params": {
      "value": 1.0     // 填充值 (默认: 1.0)
    }
  }
}
```

#### zeros (全0)
```json
{
  "_random": {
    "distribution": "zeros",
    "params": {}
  }
}
```

## 混合分布示例

### 示例1: 不同输入使用不同分布

```json
{
  "config": {
    "inputs": [
      {
        "name": "weights",
        "shape": [768, 768],
        "dtype": "float16",
        "_random": {
          "distribution": "orthogonal",
          "params": {}
        }
      },
      {
        "name": "bias",
        "shape": [768],
        "dtype": "float16",
        "_random": {
          "distribution": "zeros",
          "params": {}
        }
      },
      {
        "name": "activation",
        "shape": [512, 768],
        "dtype": "float16",
        "_random": {
          "distribution": "normal",
          "params": {
            "mean": 0.0,
            "std": 0.02
          }
        }
      }
    ]
  }
}
```

### 示例2: 稀疏注意力矩阵

```json
{
  "config": {
    "inputs": [
      {
        "name": "attention_mask",
        "shape": [128, 128],
        "dtype": "float16",
        "_random": {
          "distribution": "sparse",
          "params": {
            "density": 0.05
          }
        }
      }
    ]
  }
}
```

### 示例3: 使用CLI参数覆盖分布

```bash
# 生成使用标准正态分布的输入
python infinimetrics/utils/input_generator_cli.py \
    --config your_config.json \
    --distribution standard_normal \
    --output-json output.json

# 生成稀疏矩阵（5%密度）
python infinimetrics/utils/input_generator_cli.py \
    --config your_config.json \
    --distribution sparse \
    --density 0.05 \
    --output-json output.json

# 生成指数分布数据
python infinimetrics/utils/input_generator_cli.py \
    --config your_config.json \
    --distribution exponential \
    --scale 2.0 \
    --output-json output.json
```

## CLI参数映射

不同的分布类型对应不同的CLI参数：

| 参数 | 适用的分布 |
|------|----------|
| `--low` | uniform, randint |
| `--high` | uniform, randint |
| `--mean` | normal, lognormal |
| `--std` | normal, lognormal |
| `--scale` | uniform, exponential, laplace, cauchy |
| `--bias` | uniform |
| `--loc` | laplace, cauchy |
| `--lam` | poisson |
| `--zipf-a` | zipf |
| `--density` | sparse |
| `--sparsity` | sparse |
| `--value` | ones |

## 使用建议

1. **神经网络权重**: 使用 `orthogonal` 或 `normal` (std=0.02)
2. **偏置项**: 使用 `zeros`
3. **激活值**: 使用 `uniform` (-1, 1) 或 `normal` (0, 0.1)
4. **注意力掩码**: 使用 `sparse` (density=0.05)
5. **测试数据**: 使用 `uniform` 或 `normal`
6. **索引数据**: 使用 `randint` 或 `poisson`
7. **初始化矩阵**: 使用 `identity` 或 `orthogonal`

## 完整示例

参见 `example_mixed_distributions.json` 文件查看完整的配置示例。
