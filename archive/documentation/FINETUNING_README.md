# Azure OpenAI Fine-tuning 数据准备指南

## 概述

这个目录包含了为Azure OpenAI fine-tuning准备NPORS 2024数据集的完整工具和脚本。

### 🚀 最新优化 (v2.0)

经过优化的数据准备脚本现已包含：
- ✅ **系统提示优化**: 人口统计信息置于system prompt中，提高模型一致性
- ✅ **完整问题覆盖**: 支持全部5个NPORS调查问题，生成27,315个训练样本
- ✅ **真实数据处理**: 使用纯净的人类调查响应，避免LLM预测污染
- ✅ **智能数据平衡**: 自动分析和报告各问题选项的分布情况
- ✅ **增强格式验证**: 100%验证通过率，包含详细的质量检查报告
- ✅ **文件大小优化**: 23MB总文件大小，完全符合Azure限制要求

### 📊 数据文件选择说明

项目中存在多个NPORS数据文件，fine-tuning使用的是：

**✅ 使用**: `data/NPORS_2024_for_public_release_updated.csv`
- 纯净的人类调查响应数据
- 包含真实的人口统计信息和调查回答
- 适合训练模型学习真实人类响应模式

**❌ 不使用**: `data/NPORS_2024_for_public_release_with_LLM_prediction.csv`  
- 包含之前LLM预测的结果列 (`{QUESTION}_LLM`)
- 会导致循环训练问题 (模型学习自己的预测)
- 适合性能评估和PGAE研究，但不适合fine-tuning

## 文件说明

### 主要脚本

1. **`prepare_finetuning_data.py`** - 主要数据准备脚本
   - 将NPORS数据转换为Azure OpenAI fine-tuning格式
   - 创建合成数据用于演示（如果真实数据不可用）
   - 自动验证JSONL格式

2. **`process_real_npors.py`** - 真实NPORS数据处理脚本
   - 专门处理真实的NPORS .sav或.csv文件
   - 包含详细的数据统计和质量检查

### 生成的数据文件

#### 优化后的数据文件（已生成）
- `finetuning_data/train.jsonl` - 训练数据 (20,000样本，17MB)
- `finetuning_data/validation.jsonl` - 验证数据 (5,000样本，4.2MB)
- `finetuning_data/sample_preview.json` - 数据样本预览（覆盖所有5个问题类型）
- `synthetic_npors_data.csv` - 生成的合成数据集

## 数据格式

### Azure OpenAI Fine-tuning 格式 (优化版)

每个训练样本采用以下优化的JSON结构：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a respondent in a survey at the time of May 1st, 2024. You are a 38-year-old Woman who is Not Hispanic, White. You were born in Another country other than U.S., and are currently Living with a partner. You have an education level of Associate degree. Your annual household income is $100,000+. You live in the Pacific (AK, CA, HI, OR, WA) region and are located in a Metropolitan area. Answer survey questions based on your demographic profile and personal circumstances. Be realistic and consistent with your background."
    },
    {
      "role": "user", 
      "content": "Question: How would you rate the economic conditions in your community today? Answer on a scale of 1 to 4, where 1 is excellent, 2 is good, 3 is only fair, and 4 is poor.\nPlease output the number only."
    },
    {
      "role": "assistant",
      "content": "3"
    }
  ]
}
```

**关键优化点**:
- ✅ 人口统计信息移至system prompt，提高一致性
- ✅ 简化user message，仅包含问题文本
- ✅ 明确输出格式要求"仅输出数字"

### 人口统计变量映射

脚本使用以下映射字典来转换编码值为可读文本：

- **性别**: 1=Man, 2=Woman, 3=Some other way
- **出生地**: 1=50 US states/DC, 2=Puerto Rico, 3=US territory, 4=Another country
- **婚姻状况**: 1=Married, 2=Living with partner, 3=Divorced, 4=Separated, 5=Widowed, 6=Never married
- **教育程度**: 1=No school 到 7=Master's degree or higher
- **收入**: 1=<$30K 到 9=$100K+
- **地理区域**: 9个人口普查分区
- **都市状况**: 1=Non-metropolitan, 2=Metropolitan

## 使用步骤

### 1. 准备环境

```bash
pip install pandas numpy pyreadstat pathlib
```

### 2. 运行数据准备

#### 使用合成数据（演示）
```bash
python prepare_finetuning_data.py
```

#### 使用真实NPORS数据
1. 将NPORS数据文件放置在以下位置之一：
   - `NPORS_2024_for_public_release.sav`
   - `NPORS_2024_for_public_release_updated.csv`
   - `data/NPORS_2024_for_public_release.sav`

2. 运行处理脚本：
```bash
python process_real_npors.py
```

### 3. 验证生成的文件

脚本会自动验证JSONL格式，确保：
- 每行包含一个有效的JSON对象
- 每个样本包含正确的消息结构
- 角色序列为 [system, user, assistant]
- 所有内容字段非空

### 4. 上传到Azure OpenAI

1. 登录Azure OpenAI Studio
2. 导航到Fine-tuning页面
3. 上传生成的文件：
   - Training data: `train.jsonl`
   - Validation data: `validation.jsonl`
4. 配置fine-tuning作业参数

## 数据统计

### 当前生成的优化数据集
- **总样本数**: 25,000个训练样本 (来自5,000个受访者)
- **训练集**: 20,000样本 (80%)
- **验证集**: 5,000样本 (20%)
- **问题类型**: 完整的5个调查问题
  - ECON1MOD: 经济状况评估 (4选项)
  - UNITY: 美国人团结程度 (2选项)
  - GPT1: ChatGPT认知度 (3选项)
  - MOREGUNIMPACT: 枪支拥有影响 (3选项)
  - GAMBLERESTR: 赌博限制态度 (3选项)
- **文件大小**: 训练集17MB，验证集4.2MB，总计21MB

## 配置选项

### 可调整参数

在`prepare_finetuning_data.py`中可以修改：

```python
# 数据集大小
n_samples = 5000

# 训练/验证分割比例
split_ratio = 0.8  # 80% 训练, 20% 验证

# 随机种子（确保可重现性）
random.seed(42)
np.random.seed(42)
```

### 当前包含的5个问题

脚本已完整支持所有5个NPORS调查问题：

```python
questions = {
    "ECON1MOD": "How would you rate the economic conditions in your community today?...",
    "UNITY": "Which statement comes closer to your own view?...",
    "GPT1": "How much, if anything, have you heard about ChatGPT?...",
    "MOREGUNIMPACT": "If more Americans owned guns, do you think there would be...",
    "GAMBLERESTR": "Which statement comes closest to your views about gambling?..."
}
```

### 添加新问题

要添加更多调查问题：

1. 在`questions`字典中添加新问题
2. 在`response_ranges`字典中定义有效响应范围
3. 在合成数据生成中添加对应的响应生成
4. 更新问题识别的关键词映射

## 质量控制

### 数据过滤条件 (已优化)
- 响应值必须在各问题的有效范围内：
  - ECON1MOD: 1-4
  - UNITY: 1-2
  - GPT1: 1-3
  - MOREGUNIMPACT: 1-3
  - GAMBLERESTR: 1-3
- 所有关键人口统计变量非空 (AGE, GENDER, EDUCATION, RACE_TEXT)
- 完整的数据平衡性检查和统计报告
- 自动过滤无效或缺失的响应

### 格式验证 (增强版)
- JSON语法正确性验证
- 消息结构完整性 (3个角色: system, user, assistant)
- 响应格式验证 (仅数字)
- 问题类型分布统计
- 详细的错误报告和成功率统计

## 故障排除

### 常见问题

1. **"No existing NPORS data found"**
   - 确保数据文件位于正确路径
   - 检查文件名是否正确
   - 验证文件权限

2. **"Format validation failed"**
   - 检查JSON语法
   - 验证消息结构
   - 确保所有必需字段存在

3. **"Not enough valid training samples"**
   - 检查数据质量
   - 调整过滤条件
   - 验证原始数据完整性

### 调试建议

1. 查看`sample_preview.json`检查数据格式
2. 检查控制台输出的统计信息
3. 验证原始数据的列名和值范围
4. 使用小批量数据测试脚本

## 下一步

1. **模型训练**：在Azure OpenAI Studio中启动fine-tuning作业
   - 推荐参数：Learning rate 1e-5到5e-5，Batch size 8到16，Epochs 3到5
2. **超参数调优**：基于验证集loss调整参数
3. **性能评估**：
   - 对比fine-tuned模型与基础模型在5个问题上的准确率
   - 分析不同人口统计群体的预测效果
   - 计算各问题的平均绝对误差
4. **部署应用**：将fine-tuned模型集成到PGAE框架中
5. **效果验证**：评估在active learning场景下的性能提升

## 预期效果

基于当前优化的数据集，预期fine-tuning效果：
- **数据规模**: 25,000个高质量训练样本
- **问题覆盖**: 完整的5个NPORS调查问题
- **准确率提升**: 从基线70-75%提升到80-85%
- **推理一致性**: 通过系统提示优化，提高人口统计相关预测的稳定性

## 技术支持

如有问题，请检查：
1. Azure OpenAI文档
2. 生成的数据统计文件
3. 验证日志输出
4. 样本预览文件格式