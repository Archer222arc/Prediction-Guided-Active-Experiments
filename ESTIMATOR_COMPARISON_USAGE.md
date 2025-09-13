# 统计估计器对比工具使用指南

## 概述
`compare_estimators.py` 现在支持对多个不同预测目标进行PGAE估计器对比分析。

## 支持的预测目标

| 目标 | 描述 | 推荐方法 | 预测难度 |
|------|------|----------|----------|
| **ECON1MOD** | 经济状况评级 (1-4) | finetuned | 最难 (44.7%) |
| **UNITY** | 美国团结认知 (1-2) | baseline | 最容易 (80.9%) |
| **GPT1** | ChatGPT认知度 (1-3) | optimized | 中上 (51.1%) |
| **MOREGUNIMPACT** | 枪支管制影响 (1-3) | finetuned | 较难 (48.9%) |
| **GAMBLERESTR** | 赌博限制观点 (1-3) | finetuned | 中等 (60.1%) |

## 使用方法

### 新格式 (推荐)
```bash
# 基本使用 - 默认ECON1MOD目标
python compare_estimators.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv

# 指定不同预测目标
python compare_estimators.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv --target UNITY

# 完整参数配置
python compare_estimators.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv \
    --target GPT1 \
    --experiments 200 \
    --labels 1000 \
    --gamma 0.7 \
    --alpha 0.95 \
    --seed 123 \
    --output my_results
```

### 数据集快捷选择
- 你可以通过 `--dataset-choice` 直接选择内置数据集路径，而无需手动填写文件路径：
  - `--dataset-choice base` → `archive/predictions/NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv`
  - `--dataset-choice cot`  → `archive/predictions/NPORS_2024_cot_optimized_lr06_step560_20250911_232934.csv`

示例：
```bash
python estimators/compare_estimators.py dummy.csv --dataset-choice base --target ECON1MOD --experiments 50 --labels 500 --gamma 0.5
python estimators/compare_estimators.py dummy.csv --dataset-choice cot  --target ECON1MOD --experiments 50 --labels 500 --gamma 0.5
```
说明：若提供 `--dataset-choice`，会覆盖位置参数 `data_file`。

## CI 模式（固定CI宽度→最小标签成本）默认参数

- 入口：在 `compare_estimators.py` 里通过 `--ci-width` 启动。
- 目的：在给定置信区间宽度目标下，比较各方法达到该宽度所需的最少真实标签数（越少越省成本）。

已部署的调优默认值（基于 ECON1MOD，gamma=0.5 的调参结果）：

- PGAE（仅 CI 模式生效）：
  - `gamma`: 0.5（可通过 `--gamma-grid` 覆盖）
  - 设计参数：`min_var_threshold=1e-4`, `prob_clip_min=0.1`, `prob_clip_max=0.9`
  - CI估计参数（RF/CV）：`n_estimators_mu=100`, `n_estimators_tau=200`, `K=5`, `max_depth=10`
- Adaptive PGAE（CI 模式与常规模式一致）：
  - `gamma`: 0.5
  - `batch_size`: 250
  - `design_update_freq`: 1
  - `warmup_batches`: 2

示例：

```bash
# 在 ECON1MOD 上比较达到 CI<=0.10 的最小标签数（默认并发）
python compare_estimators.py archive/predictions/NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv \
  --target ECON1MOD \
  --ci-width 0.10 \
  --methods PGAE Adaptive_PGAE Active_Inference Naive \
  --gamma-grid 0.5 \
  -a 0.95 --concurrent --max-workers 10
```

说明：
- 如不希望采用上述 PGAE 的默认调优参数，可自行在源码中覆盖或使用 `estimators/tune_ci_pgae.py` 进行参数扫描。
- 该默认值来自调参结果：在 n_labels≈1000 附近，PGAE 的平均CI宽度≈0.1000、覆盖率≈0.95、MSE≈0.00064；Adaptive 在 `batch_size=250, warmup=2` 表现最佳（示例）。

### 旧格式 (兼容)
```bash
# 仍然支持旧格式，默认使用ECON1MOD
python compare_estimators.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv 100 500
```

## 参数说明

### 必需参数
- `data_file`: 数据文件路径

### 可选参数
- `--target/-t`: 预测目标 (默认: ECON1MOD)
  - 选择: ECON1MOD, UNITY, GPT1, MOREGUNIMPACT, GAMBLERESTR
- `--experiments/-e`: 实验次数 (默认: 100)
- `--labels/-l`: 每次实验的真实标签数量 (默认: 500)
- `--gamma/-g`: PGAE gamma参数 (默认: 0.5)
- `--alpha/-a`: 置信水平 (默认: 0.90)
- `--seed/-s`: 随机种子 (默认: 42)
- `--output/-o`: 输出文件名前缀

## 输出文件

### 结果文件
- 格式: `estimator_comparison_{target}_{timestamp}.json`
- 例如: `estimator_comparison_unity_20250912_143022.json`

### 可视化文件
- 文件名: `estimator_comparison.png`
- 包含性能对比图表

### 比较运行日志（CSV，便于累计对比）
- 文件名: `compare_runs_log.csv`（可用 `--results-csv` 指定）
- 行结构:
  - `mode`: `mse-compare` 或 `ci-cost`
  - 公共设置: `timestamp`, `target`, `dataset`, `alpha`, `experiments`, `method`, `gamma`, `settings_key`
  - MSE模式: `n_true_labels`, `mse`, `avg_ci_length`, `coverage_rate`, `execution_time`
  - CI成本模式: `ci_width`, `ci_tolerance`, `required_labels`, `avg_ci_length`, `mse_snapshot`, `coverage_snapshot`
- 去重覆盖: 若同一 `settings_key` 和 `method` 再次运行，将覆盖旧行

## 使用示例

### 1. 比较不同任务的估计器性能
```bash
# 经济任务 (最难预测)
python compare_estimators.py data.csv --target ECON1MOD --experiments 500

# 团结任务 (最容易预测)
python compare_estimators.py data.csv --target UNITY --experiments 500

# AI任务 (中等难度)
python compare_estimators.py data.csv --target GPT1 --experiments 500
```

### 2. 参数敏感性分析
```bash
# 不同gamma值测试
python compare_estimators.py data.csv --target ECON1MOD --gamma 0.3 --output gamma_03
python compare_estimators.py data.csv --target ECON1MOD --gamma 0.7 --output gamma_07

# 不同样本数测试
python compare_estimators.py data.csv --target ECON1MOD --labels 300 --output labels_300
python compare_estimators.py data.csv --target ECON1MOD --labels 1000 --output labels_1000
```

### 3. 高精度实验
```bash
# 大规模实验
python compare_estimators.py data.csv \
    --target ECON1MOD \
    --experiments 1000 \
    --labels 1000 \
    --alpha 0.95 \
    --output high_precision
```

## 预期结果

### 输出示例
```
✅ ECON1MOD 统计估计器对比完成!
预测目标: ECON1MOD
实验设置: 100 experiments, 500 labels per experiment
结果文件: estimator_comparison_econ1mod_20250912_143022.json
可视化文件: estimator_comparison.png

🏆 最佳方法: PGAE
   MSE: 0.001234
   覆盖率: 0.9200
```

### JSON结果格式
```json
{
  "PGAE": {
    "mse": 0.001234,
    "bias": -0.000123,
    "variance": 0.001111,
    "avg_ci_length": 0.1394,
    "coverage_rate": 0.9200,
    "execution_time": 45.67,
    "n_experiments": 100
  },
  "Adaptive_PGAE": { ... },
  "Active_Inference": { ... },
  "summary": {
    "true_value": 2.8456,
    "n_experiments": 100,
    "n_true_labels": 500,
    "gamma": 0.5,
    "total_execution_time": 120.34
  }
}
```

## 注意事项

1. **数据要求**: 数据文件必须包含对应的预测列 (如 `UNITY_LLM`, `GPT1_LLM` 等)
2. **内存使用**: 大规模实验可能需要较大内存，建议逐步增加实验规模
3. **并发执行**: 所有estimator都使用并发优化，提升执行速度
4. **结果解释**: 不同任务的预测难度不同，MSE绝对值不能直接跨任务比较

## 故障排除

### 常见错误
1. **KeyError**: 检查数据文件是否包含所需的预测列
2. **ValueError**: 检查目标名称是否正确拼写
3. **MemoryError**: 减少experiments或labels数量

### 调试建议
```bash
# 小规模测试
python compare_estimators.py data.csv --target ECON1MOD --experiments 10 --labels 50

# 检查数据列
python -c "import pandas as pd; print(pd.read_csv('data.csv').columns.tolist())"
```

---
*更新时间: 2025-09-12*
