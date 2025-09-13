# 数据处理脚本使用指南

## 概述

`data_processor.py` 是从 `tianyi_digital_twin.ipynb` 中提取的独立数据处理脚本，用于自动化下载和处理数字孪生数据集。

## 功能特性

- 🔄 **自动下载**: 从Hugging Face自动下载LLM-Digital-Twin/Twin-2K-500数据集
- 🛡️ **错误处理**: 完善的异常处理和缓存清理机制
- 📊 **数据验证**: 自动验证数据完整性和格式正确性
- 💾 **格式转换**: 将数据保存为标准JSON格式
- 📈 **样本分析**: 提供personas数据的统计分析

## 使用方法

### 基本用法

```bash
# 下载默认30个personas和ground truth数据
python data_processor.py

# 指定persona数量
python data_processor.py --num-personas 100

# 自定义输出目录
python data_processor.py --output-dir ./custom_data

# 强制重新下载
python data_processor.py --force-reload

# 跳过ground truth数据
python data_processor.py --skip-ground-truth

# 仅分析现有数据
python data_processor.py --analyze-only
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-personas` | int | 30 | 要下载的persona数量 |
| `--output-dir` | str | ./data | 数据输出目录 |
| `--force-reload` | flag | False | 强制清理缓存并重新下载 |
| `--skip-ground-truth` | flag | False | 跳过ground truth数据下载 |
| `--analyze-only` | flag | False | 仅分析现有数据，不下载新数据 |

## 输出文件

### 文件结构
```
data/
├── personas_output.json      # Persona档案数据
└── ground_truth_output.json  # 真实人类响应数据
```

### 数据格式

#### personas_output.json
```json
{
  "pid_1": "The following is a description of a person...",
  "pid_2": "The following is a description of a person...",
  ...
}
```

#### ground_truth_output.json
```json
[
  {
    "ElementType": "Block",
    "BlockName": "False consensus",
    "Questions": [...],
    "Answers": {...}
  },
  ...
]
```

## 使用示例

### 示例1: 快速开始
```bash
# 下载并处理数据
python data_processor.py --num-personas 50

# 输出示例:
# 2024-01-01 10:00:00 - INFO - 正在加载 50 个persona摘要...
# 2024-01-01 10:00:05 - INFO - ✅ 成功加载 50 个personas
# 2024-01-01 10:00:10 - INFO - ✅ 成功加载 2058 条ground truth记录
# 2024-01-01 10:00:12 - INFO - ✅ 数据验证通过
```

### 示例2: 自定义配置
```bash
# 大量数据下载到指定目录
python data_processor.py \
    --num-personas 200 \
    --output-dir ./experiments/batch_001 \
    --force-reload
```

### 示例3: 分析现有数据
```bash
# 分析已下载的数据
python data_processor.py --analyze-only

# 输出示例:
# ==================================================
# PERSONA 样本分析
# ==================================================
# 总persona数量: 30
# 
# 【pid_1】
# 摘要长度: 1234 字符
# 前500字符: The following is a description of a person...
# 关键信息: {'Gender': 'Male', 'Age': '18-29', ...}
```

## 错误处理

### 常见问题及解决方案

#### 1. 网络连接问题
```bash
# 症状: 下载超时或连接失败
# 解决: 检查网络连接，重试下载
python data_processor.py --force-reload
```

#### 2. 缓存损坏
```bash
# 症状: 数据加载异常
# 解决: 清理缓存重新下载
python data_processor.py --force-reload
```

#### 3. 磁盘空间不足
```bash
# 症状: 保存文件失败
# 解决: 清理磁盘空间，选择其他输出目录
python data_processor.py --output-dir /path/to/larger/disk
```

## 集成使用

### 在其他脚本中使用
```python
from data_processor import load_personas, load_ground_truth_data

# 加载数据
personas = load_personas(num_personas=100)
ground_truth = load_ground_truth_data()

# 后续处理...
```

### 与原notebook的对应关系
```python
# 原notebook代码                    对应的脚本函数
# load_personas(NUM_PERSONAS)      → load_personas()
# wave_split = load_dataset(...)    → load_ground_truth_data()
# save_ground_truth_to_json(...)   → save_ground_truth_to_json()
```

## 性能优化

### 批量处理建议
```bash
# 小批量测试 (快速验证)
python data_processor.py --num-personas 10

# 中等批量 (开发调试)
python data_processor.py --num-personas 100

# 大批量 (完整数据集)
python data_processor.py --num-personas 2000
```

### 存储优化
- Ground truth文件较大 (~3M行)，可使用 `--skip-ground-truth` 跳过
- 对于实验场景，建议先用小量数据验证流程

## 日志分析

脚本提供详细的运行日志：
- ✅ 成功操作 (绿色勾号)
- ⚠️  警告信息 (感叹号)
- ❌ 错误信息 (红色X)

### 日志示例
```
2024-01-01 10:00:00 - INFO - 正在加载 30 个persona摘要...
2024-01-01 10:00:02 - INFO - ✅ 数据集加载成功
2024-01-01 10:00:03 - INFO - ✅ 成功加载 30 个personas
2024-01-01 10:00:05 - INFO - 正在保存ground truth数据到 ground_truth_output.json...
2024-01-01 10:00:08 - INFO - ✅ 已保存到 ./data/ground_truth_output.json
2024-01-01 10:00:09 - INFO - ✅ 数据验证通过
2024-01-01 10:00:10 - INFO - 🎉 数据处理完成!
```

---

此脚本将notebook中的数据处理逻辑标准化，便于在不同实验环境中重复使用和自动化执行。