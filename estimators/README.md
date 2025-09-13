# Statistical Estimators for Survey Prediction

本目录包含了基于LLM_prediction.ipynb实现的三种统计估计器，用于调查响应预测的主动实验设计。

## 概述

这些估计器实现了PGAE（Prediction-Guided Active Experiments）框架下的不同策略，旨在高效地使用LLM预测来估计调查响应的总体参数。

## 估计器类型

### 1. PGAE估计器 (`pgae_estimator.py`)
- **原理**: 使用预测引导的主动实验设计
- **特点**: 根据预测不确定性优化实验设计，固定设计参数
- **适用场景**: 当有充足先验信息时的稳定实验设计

### 2. 自适应PGAE估计器 (`adaptive_pgae_estimator.py`)
- **原理**: 动态调整实验设计参数
- **特点**: 在实验过程中根据观察到的数据更新设计
- **适用场景**: 先验信息不足，需要在线学习的场景

### 3. 主动统计推断估计器 (`active_inference_estimator.py`)
- **原理**: 基于预测误差的主动采样
- **特点**: 重点关注预测误差较大的样本
- **适用场景**: 当预测质量不均匀时的效率优化

## 核心参数

- **X**: 协变量列表（如`['EDUCATION']`）
- **F**: LLM预测列名（如`'ECON1MOD_LLM'`）
- **Y**: 真实标签列名（如`'ECON1MOD'`）
- **gamma**: 控制实验强度的参数（0-1之间）

## 使用方法

### 单独运行估计器

```bash
# PGAE估计器
python pgae_estimator.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv

# 自适应PGAE估计器
python adaptive_pgae_estimator.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv

# 主动统计推断估计器
python active_inference_estimator.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv
```

### 对比所有估计器

```bash
# 运行对比实验
python compare_estimators.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv

# 指定实验次数和标签数量
python compare_estimators.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv 100 500
```

## 文件结构

```
estimators/
├── utils.py                           # 工具函数
├── pgae_estimator.py                  # PGAE估计器
├── adaptive_pgae_estimator.py         # 自适应PGAE估计器
├── active_inference_estimator.py      # 主动统计推断估计器
├── compare_estimators.py              # 估计器对比工具
└── README.md                          # 说明文档
```

## 输出结果

每个估计器都会生成包含以下指标的结果：

- **MSE**: 均方误差
- **Bias**: 偏差
- **Variance**: 方差
- **Coverage Rate**: 置信区间覆盖率
- **Average CI Length**: 平均置信区间长度
- **Execution Time**: 执行时间

## 实验设计原理

### PGAE框架
PGAE方法通过以下步骤优化实验设计：

1. **预测阶段**: 使用LLM对所有样本进行预测
2. **设计阶段**: 根据预测不确定性计算采样和实验概率
3. **实验阶段**: 按设计概率进行主动采样和标记
4. **估计阶段**: 使用加权方法估计总体参数

### 关键概念

- **接受概率** (`accept_prob`): 样本被选入实验的概率
- **实验概率** (`exp_prob`): 被选中样本获得真实标签的概率
- **采样PMF** (`sample_pmf`): 基于不确定性调整的采样分布

### 数学基础

估计器基于以下数学框架：

```
τ̂ = E[F + (Y - F) × I / p]
```

其中：
- F: LLM预测值
- Y: 真实值
- I: 标记指示器
- p: 实验概率

## 性能对比

根据notebook中的实验结果：

| 方法 | MSE | CI长度 | 覆盖率 |
|------|-----|--------|--------|
| PGAE | 0.0013 | 0.1394 | 0.999 |
| Adaptive PGAE | 0.0032 | 0.1180 | 0.642 |
| Active Inference | 0.0015 | 0.1572 | 0.959 |

**建议使用顺序**:
1. **PGAE**: 最稳定的方法，适合大多数场景
2. **Active Inference**: 中等性能，实现简单
3. **Adaptive PGAE**: 虽然理论上更灵活，但实际性能可能不稳定

## 依赖要求

```python
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
tqdm
```

## 注意事项

1. **数据格式**: 确保数据包含所需的协变量、预测列和真实标签列
2. **参数调优**: gamma参数对性能影响很大，建议在0.3-0.7范围内调整
3. **计算资源**: 实验次数越多结果越稳定，但计算时间也会增加
4. **随机种子**: 为了结果可重现，建议设置固定的随机种子

## 扩展使用

这些估计器可以轻松扩展到其他预测任务：

```python
# 示例：扩展到多个协变量
estimator = PGAEEstimator(
    X=['EDUCATION', 'AGE', 'INCOME'],  # 多个协变量
    F='PREDICTION_COL',
    Y='TRUE_LABEL_COL',
    gamma=0.5
)

# 示例：批量测试不同参数
for gamma in [0.3, 0.5, 0.7]:
    estimator = PGAEEstimator(X, F, Y, gamma)
    results = estimator.run_experiments(df, n_experiments=100)
    print(f"Gamma {gamma}: MSE = {results['mse']:.6f}")
```

## 引用

如果使用这些估计器，请引用相关的PGAE论文和本项目。