# NPORS Fine-tuning 工作流程

## 概述
这是一套完整的 NPORS 2024 数据集微调 GPT-4.1-mini 模型的脚本集合。

## 工作流程

### 第一步：数据准备
```bash
python 1_prepare_npors_finetuning_data.py
```
- 处理真实 NPORS 2024 数据集
- 生成微调格式的训练和验证文件
- 输出：`finetuning_data/train.jsonl` 和 `finetuning_data/validation.jsonl`

### 第二步：创建微调作业
```bash
python 2_azure_gpt41_mini_finetuning.py create
```
- 上传训练数据到 Azure OpenAI
- 创建 GPT-4.1-mini 微调作业
- 使用 North Central US 区域 (支持 gpt-4.1-mini 微调)
- 输出：作业信息保存到 `gpt41_mini_job_info.json`

### 第三步：监控进度
```bash
# 方式1：持续监控
python 2_azure_gpt41_mini_finetuning.py monitor

# 方式2：快速状态检查
python 3_monitor_finetuning_status.py
```

## 文件说明

### 核心脚本
- `1_prepare_npors_finetuning_data.py` - 数据准备脚本
- `2_azure_gpt41_mini_finetuning.py` - 主要微调脚本
- `3_monitor_finetuning_status.py` - 状态监控脚本

### 配置文件
- `config/azure_models_config.json` - Azure 端点和API密钥配置

### 数据文件
- `data/` - 原始 NPORS 2024 数据
- `finetuning_data/` - 处理后的微调数据
- `gpt41_mini_job_info.json` - 微调作业信息
- `gpt41_mini_final_result.json` - 最终结果（完成后）

## 微调参数
- 模型：`gpt-4.1-mini-2025-04-14`
- 训练轮数：3 epochs
- 批次大小：1
- 学习率倍数：1.0
- 数据量：约 27,315 个训练样本

## 当前状态
- 微调作业ID：`ftjob-a4f068c459104484b183fa8512d71379`
- 状态：pending (排队等待处理)
- 区域：North Central US

## 注意事项
1. 确保使用 North Central US 区域的 Azure OpenAI 资源
2. gpt-4.1-mini 只在特定区域支持微调
3. 文件上传后需要等待处理完成才能创建作业
4. 微调可能需要几分钟到几小时完成