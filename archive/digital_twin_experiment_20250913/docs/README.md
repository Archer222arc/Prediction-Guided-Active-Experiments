# Digital Twin Experiment Archive - 2025年9月13日

## 实验概述
本归档包含2025年9月13日完成的数字双胞胎LLM预测实验的完整内容。该实验专注于使用GPT-5-Mini和思维链提示来改进基于角色的政策意见预测。

## 核心功能实现
1. **Azure OpenAI集成**: 配置archer222arc端点与GPT-5-Mini部署
2. **角色化提示系统**: 使用persona parquet数据的增强预测系统
3. **NPORS列名标准化**: 统一列命名(CARBONTAX, CLEANENERGY等)
4. **双胞胎对比系统**: MSE和CI宽度成本分析框架
5. **批处理脚本**: 统一参数格式的自动化实验执行
6. **有放回采样**: 支持标签数>数据集大小的稳健采样

## 归档目录结构
```
digital_twin_experiment_20250913/ (总计21个有效文件, 1.7MB)
├── scripts/     (5个文件) - 核心处理和预测脚本
├── estimators/  (1个文件) - 双胞胎比较估计器
├── data/        (6个文件, ~120KB) - 生成的预测数据集和工具
├── results/     (8个文件, ~1.4MB) - 实验输出(MSE, CI分析)
└── docs/        (1个文件) - 本文档
```

## 详细文件清单

### 核心处理脚本 (`scripts/` 目录)
1. **`digital_twin_prediction.py`** - Azure OpenAI集成的主预测脚本
   - 配置archer222arc端点, GPT-5-Mini模型
   - 角色化提示系统, 支持CoT和Base模式

2. **`process_ground_truth_predictions.py`** - 真值数据处理脚本
   - 从persona parquet提取教育信息
   - NPORS格式列名标准化

3. **`merge_predictions_npors.py`** - 预测结果与真值合并脚本
   - 生成NPORS格式的最终数据集

4. **`run_twin_compare_mse_all.sh`** - MSE对比批处理脚本
   - 支持CoT/Base数据集选择
   - 统一--参数格式

5. **`run_twin_compare_ci_all.sh`** - CI宽度成本分析批处理脚本
   - 二分搜索优化的标签数量分析

### 估计器 (`estimators/` 目录)
1. **`twin_compare.py`** - 增强的双胞胎比较工具
   - 支持MSE和CI宽度两种模式
   - 4种估计方法: PGAE, Adaptive PGAE, Active Inference, Naive
   - 修复无限循环bug, 添加CSV输出功能

### 数据文件 (`data/` 目录, 总计~120KB)
1. **`enhanced_digital_twin_predictions_base_20250913_122146.csv`** (57KB)
   - Base模式预测结果, 1000条记录

2. **`enhanced_digital_twin_predictions_cot_20250913_132133.csv`** (57KB)
   - Chain-of-Thought模式预测结果, 1000条记录

3. **`enhanced_digital_twin_predictions_*_stats.json`** (2个文件)
   - 预测结果统计信息

4. **`convert_personas_to_parquet.py`** - 角色数据转换工具
5. **`extract_personas.py`** - JSON角色数据提取工具

### 实验结果 (`results/` 目录, 总计~1.4MB)
#### MSE对比结果:
1. **`20250913_141453_twin_compare_mse.csv`** (13KB) - MSE对比摘要
2. **`20250913_141453_twin_compare_runs.csv`** (692KB) - 详细运行记录
3. **`20250913_142403_twin_compare_mse.csv`** (13KB) - 第二轮MSE对比摘要
4. **`20250913_142403_twin_compare_runs.csv`** (688KB) - 第二轮详细记录

#### CI宽度分析结果:
5. **`20250913_150952_twin_ci_summary.csv`** (4.5KB) - CI分析摘要
6. **`20250913_150952_twin_ci_audit.csv`** (24KB) - CI搜索审计轨迹
7. **`20250913_152326_twin_ci_summary.csv`** (4.6KB) - 第二轮CI摘要
8. **`20250913_152326_twin_ci_audit.csv`** (25KB) - 第二轮审计轨迹

## 技术实现详情

### Azure配置
- **端点**: archer222arc
- **模型**: gpt-5-mini
- **API版本**: 2024-12-01-preview
- **部署名称**: gpt-5-mini

### NPORS列名标准化
实现了统一的政策问题列名映射:
```python
NPORS_QUESTIONS = [
    'CARBONTAX',    # 碳税
    'CLEANENERGY',  # 清洁能源
    'CLEANELEC',    # 清洁电力
    'MEDICAREALL',  # 全民医保
    'PUBLICOPTION', # 公共选择
    'IMMIGRATION',  # 移民政策
    'FAMILYLEAVE',  # 家庭假期
    'WEALTHTAX',    # 财富税
    'DEPORTATIONS', # 驱逐出境
    'MEDICVOUCHER'  # 医疗券
]
```

### 估计器方法
- **PGAE**: 预测引导的主动实验
- **Adaptive PGAE**: 自适应PGAE
- **Active Inference**: 主动推断
- **Naive**: 朴素基准方法

### 关键Bug修复
1. **属性错误修复**: `response_ranges` → `valid_responses`
2. **无限循环修复**: CI宽度搜索达到max_labels时的循环问题
3. **CSV输出增强**: 为CI宽度分析模式添加CSV输出功能
4. **参数格式统一**: 所有批处理脚本统一使用--选项格式

## 使用示例

### MSE对比分析
```bash
./scripts/run_twin_compare_mse_all.sh --dataset cot --experiments 50 --labels 600
```

### CI宽度成本分析
```bash
./scripts/run_twin_compare_ci_all.sh --dataset cot --ci-width 0.15 --max-labels 2000
```

## 实验成果总结
- ✅ **数据处理**: 真值数据处理，100%角色ID精确匹配
- ✅ **CI分析**: 实现了稳健的CI宽度成本分析与二分搜索优化
- ✅ **对比实验**: 生成CoT和Base数据集的全面对比结果
- ✅ **采样验证**: 验证有放回采样正确处理标签数>数据集大小的情况
- ✅ **架构完善**: 建立了完整的数字双胞胎预测分析框架

## 数据集规模
- **Base模式**: 1000个角色 × 10个政策问题 = 10,000个预测
- **CoT模式**: 1000个角色 × 10个政策问题 = 10,000个预测
- **实验运行**: MSE模式50次实验，CI模式最多2000标签搜索

---
**日期**: 2025年9月13日
**状态**: 实验完成并归档
**总文件**: 21个有效文件, 约1.7MB