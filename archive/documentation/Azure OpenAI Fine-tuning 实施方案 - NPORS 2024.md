# Azure OpenAI Fine-tuning 实施方案 - NPORS 2024

## 一、项目背景

基于现有NPORS 2024数据集（5,626个样本）和已实现的三种提示策略，通过fine-tuning进一步提升预测准确率。

### 现有基础
- **数据集**: `NPORS_2024_for_public_release.sav` (已转换为CSV)
- **已实现策略**: Basic Prompting, Chain-of-Thought, Few-Shot
- **当前模型**: Azure OpenAI "o4-mini" (GPT-4变体)
- **基线准确率**: ~70-75%

## 二、数据准备

### 2.1 利用现有数据结构

```python
import pandas as pd
import json
import numpy as np

# 读取已处理的数据
df = pd.read_csv('NPORS_2024_for_public_release_updated.csv')

# 使用现有的映射字典
race_labels = {
    1: "White", 2: "Black", 3: "Asian",
    4: "American Indian or Alaska Native",
    5: "Native Hawaiian or Other Pacific Islander", 
    6: "Other"
}

gender_map = {1: "Man", 2: "Woman", 3: "Some other way"}
birthplace_map = {
    1: "50 U.S. states or D.C.", 
    2: "Puerto Rico",
    3: "U.S. territory", 
    4: "Another country other than U.S."
}
marital_map = {
    1: "Married", 2: "Living with a partner", 
    3: "Divorced", 4: "Separated", 
    5: "Widowed", 6: "Never married"
}
education_map = {
    1: "No school", 
    2: "Kindergarten to grade 11", 
    3: "High school graduate",
    4: "Some college, no degree", 
    5: "Associate degree",
    6: "Bachelor's degree", 
    7: "Master's degree or higher"
}
division_map = {
    1: "New England", 2: "Middle Atlantic",
    3: "East North Central", 4: "West North Central",
    5: "South Atlantic", 6: "East South Central",
    7: "West South Central", 8: "Mountain", 9: "Pacific"
}
income_map = {
    1: "<$30,000", 2: "$30,000-39,999", 
    3: "$40,000-49,999", 4: "$50,000-59,999",
    5: "$60,000-69,999", 6: "$70,000-79,999", 
    7: "$80,000-89,999", 8: "$90,000-99,999", 
    9: "$100,000+"
}
metro_map = {1: "Non-metropolitan", 2: "Metropolitan"}
```

### 2.2 生成Fine-tuning训练数据

```python
def create_training_data(df):
    """将NPORS数据转换为Azure fine-tuning格式"""
    
    training_samples = []
    
    # 定义问题文本
    questions = {
        "ECON1MOD": "How would you rate the economic conditions in your community today? (1=Excellent, 2=Good, 3=Only fair, 4=Poor)",
        "UNITY": "Which statement comes closer to your view? (1=Americans are united, 2=Americans are divided)",
        "GPT1": "How much have you heard about ChatGPT? (1=A lot, 2=A little, 3=Nothing at all)",
        "MOREGUNIMPACT": "If more Americans owned guns, there would be... (1=More crime, 2=Less crime, 3=No difference)",
        "GAMBLERESTR": "Gambling restrictions should be... (1=MORE restrictions, 2=About right, 3=FEWER restrictions)"
    }
    
    for idx, row in df.iterrows():
        # 构建人口统计学描述（使用现有格式）
        demographic_prompt = f"""You are a respondent in a survey at the time of May 1st, 2024.
You are a {row['AGE']}-year-old {gender_map.get(row['GENDER'], 'Unknown')} who is {row.get('RACE_TEXT', 'Unknown')}.
You were born in {birthplace_map.get(row['BIRTHPLACE'], 'Unknown')}, and are currently {marital_map.get(row['MARITAL'], 'Unknown')}.
You have an education level of {education_map.get(row['EDUCATION'], 'Unknown')}.
Your annual household income is {income_map.get(row['INC_SDT1'], 'Unknown')}.
You live in the {division_map.get(row['DIVISION'], 'Unknown')} region and are located in a {metro_map.get(row['METRO'], 'Unknown')} area."""
        
        # 为每个问题创建训练样本
        for qid, qtext in questions.items():
            if pd.notna(row.get(qid)):
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": demographic_prompt
                        },
                        {
                            "role": "user",
                            "content": f"Question: {qtext}\nPlease output the number only."
                        },
                        {
                            "role": "assistant",
                            "content": str(int(row[qid]))
                        }
                    ]
                }
                training_samples.append(sample)
    
    return training_samples

# 生成训练数据
training_data = create_training_data(df)

# 分割训练集和验证集
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(
    training_data, 
    test_size=0.2, 
    random_state=42
)

# 保存为JSONL格式
with open('npors_train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('npors_validation.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

print(f"训练样本数: {len(train_data)}")
print(f"验证样本数: {len(val_data)}")
```

## 三、Azure API配置

### 3.1 使用现有配置

```python
from openai import AzureOpenAI

# 使用项目现有的配置
endpoint = "https://chenh-m9vmscvr-eastus2.cognitiveservices.azure.com/"
api_key = "YOUR_API_KEY"  # 从config.json读取
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)
```

### 3.2 上传训练文件

```python
# 上传训练文件
def upload_file(filepath, purpose="fine-tune"):
    with open(filepath, "rb") as f:
        response = client.files.create(
            file=f,
            purpose=purpose
        )
    return response.id

train_file_id = upload_file("npors_train.jsonl")
val_file_id = upload_file("npors_validation.jsonl")

print(f"训练文件ID: {train_file_id}")
print(f"验证文件ID: {val_file_id}")
```

## 四、执行Fine-tuning

### 4.1 创建Fine-tuning任务

```python
# 创建fine-tuning任务
fine_tune_job = client.fine_tuning.jobs.create(
    training_file=train_file_id,
    validation_file=val_file_id,
    model="gpt-4o-mini",  # 基础模型
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 8,
        "learning_rate_multiplier": 0.1
    },
    suffix="npors-2024"
)

job_id = fine_tune_job.id
print(f"Fine-tuning任务ID: {job_id}")
```

### 4.2 监控训练进度

```python
import time

def monitor_job(job_id):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"状态: {job.status}")
        
        if job.status == "succeeded":
            print(f"训练完成！")
            print(f"Fine-tuned模型: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status == "failed":
            print(f"训练失败: {job.error}")
            return None
        
        # 显示最新事件
        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=5
        )
        for event in events.data:
            print(f"  {event.created_at}: {event.message}")
        
        time.sleep(60)

# 监控训练
fine_tuned_model = monitor_job(job_id)
```

## 五、模型测试与比较

### 5.1 使用Fine-tuned模型

```python
def predict_with_finetuned(row, question, model_name):
    """使用fine-tuned模型预测"""
    
    # 构建与训练时相同的prompt
    demographic_prompt = f"""You are a respondent in a survey at the time of May 1st, 2024.
You are a {row['AGE']}-year-old {gender_map.get(row['GENDER'])} who is {row.get('RACE_TEXT')}.
You were born in {birthplace_map.get(row['BIRTHPLACE'])}, and are currently {marital_map.get(row['MARITAL'])}.
You have an education level of {education_map.get(row['EDUCATION'])}.
Your annual household income is {income_map.get(row['INC_SDT1'])}.
You live in the {division_map.get(row['DIVISION'])} region and are located in a {metro_map.get(row['METRO'])} area."""
    
    response = client.chat.completions.create(
        model=model_name,  # 使用fine-tuned模型
        messages=[
            {"role": "system", "content": demographic_prompt},
            {"role": "user", "content": f"Question: {question}\nPlease output the number only."}
        ],
        temperature=0.3,  # 降低随机性
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip()

# 测试预测
test_row = df.iloc[0]
prediction = predict_with_finetuned(
    test_row, 
    "How would you rate the economic conditions in your community today?",
    fine_tuned_model
)
print(f"预测结果: {prediction}")
```

### 5.2 性能对比评估

```python
def evaluate_models(test_df, n_samples=100):
    """对比不同方法的性能"""
    
    results = {
        'basic': {'correct': 0, 'total': 0},
        'cot': {'correct': 0, 'total': 0},
        'few_shot': {'correct': 0, 'total': 0},
        'fine_tuned': {'correct': 0, 'total': 0}
    }
    
    # 随机采样测试
    test_sample = test_df.sample(n=min(n_samples, len(test_df)))
    
    for idx, row in test_sample.iterrows():
        for question in ['ECON1MOD', 'UNITY', 'GPT1']:
            if pd.notna(row[question]):
                actual = str(int(row[question]))
                
                # 测试fine-tuned模型
                pred_ft = predict_with_finetuned(row, questions[question], fine_tuned_model)
                results['fine_tuned']['total'] += 1
                if pred_ft == actual:
                    results['fine_tuned']['correct'] += 1
                
                # 对比其他方法（如果有历史结果）
                if f'{question}_LLM' in row:
                    pred_basic = str(int(row[f'{question}_LLM']))
                    results['basic']['total'] += 1
                    if pred_basic == actual:
                        results['basic']['correct'] += 1
    
    # 计算准确率
    for method in results:
        if results[method]['total'] > 0:
            acc = results[method]['correct'] / results[method]['total']
            print(f"{method}: {acc:.2%} ({results[method]['correct']}/{results[method]['total']})")
    
    return results

# 执行评估
evaluation_results = evaluate_models(df)
```

## 六、成本优化策略

### 6.1 混合策略（推荐）

```python
class HybridPredictor:
    """结合fine-tuned和few-shot的混合预测器"""
    
    def __init__(self, fine_tuned_model, base_model="o4-mini"):
        self.fine_tuned_model = fine_tuned_model
        self.base_model = base_model
        self.cache = {}
    
    def predict(self, row, question, use_cache=True):
        # 生成缓存key
        cache_key = f"{row['AGE']}_{row['GENDER']}_{row['EDUCATION']}_{row['INC_SDT1']}_{question}"
        
        # 检查缓存
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 对于高置信度场景使用fine-tuned模型
        if self._is_high_confidence_scenario(row):
            result = self._predict_finetuned(row, question)
        else:
            # 对于复杂场景使用few-shot
            result = self._predict_fewshot(row, question)
        
        # 缓存结果
        if use_cache:
            self.cache[cache_key] = result
        
        return result
    
    def _is_high_confidence_scenario(self, row):
        """判断是否为高置信度场景"""
        # 教育程度高且收入稳定的群体预测更准确
        return (row['EDUCATION'] >= 5 and row['INC_SDT1'] >= 4)
```

### 6.2 批处理优化

```python
def batch_predict(df, model_name, batch_size=20):
    """批量预测以提高效率"""
    
    predictions = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_messages = []
        
        for _, row in batch.iterrows():
            demographic_prompt = build_demographic_prompt(row)
            batch_messages.append({
                "custom_id": f"row_{row.name}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": demographic_prompt},
                        {"role": "user", "content": "Rate economic conditions (1-4):"}
                    ],
                    "max_tokens": 10
                }
            })
        
        # 使用批处理API（50%折扣）
        # 注意：需要使用Azure Batch API endpoint
        # batch_response = client.batches.create(input_file_id=batch_file_id)
        
    return predictions
```

## 七、实施计划

### 第1周：数据准备与测试
- [ ] 生成训练数据（1天）
- [ ] 小规模测试（500样本）验证格式（1天）
- [ ] 上传文件并启动测试训练（1天）
- [ ] 评估测试结果（2天）

### 第2周：全量训练
- [ ] 准备完整训练集（2天）
- [ ] 执行fine-tuning（2-3天）
- [ ] 模型评估（2天）

### 第3周：优化与部署
- [ ] 实施缓存机制（2天）
- [ ] 批处理优化（1天）
- [ ] 性能测试（2天）

## 八、预期成果

### 性能指标
- **基线（现有方法）**: 70-75%准确率
- **Fine-tuned模型**: 80-85%准确率
- **混合策略**: 85-90%准确率

### 成本估算
```
训练成本（一次性）:
- 训练数据: ~20,000样本 × 200 tokens = 4M tokens
- 训练成本: ~$50-100

推理成本（月度）:
- Fine-tuned模型: $0.15/1M输入 + $0.60/1M输出
- 月度预算: $3,000可处理~20M tokens
- 缓存命中后: 有效处理量提升3-5倍
```

## 九、注意事项

1. **数据质量**: 确保RACE_TEXT字段正确生成
2. **模型版本**: 记录fine-tuned模型ID用于版本控制
3. **渐进实施**: 先用500样本测试，验证后再全量训练
4. **保留基线**: 保持现有方法作为fallback方案
5. **监控指标**: 追踪各人口群体的预测公平性

## 十、快速启动代码

```python
# 完整的快速启动脚本
def quick_start_finetuning():
    # 1. 加载数据
    df = pd.read_csv('NPORS_2024_for_public_release_updated.csv')
    
    # 2. 生成训练数据
    training_data = create_training_data(df)
    
    # 3. 保存JSONL文件
    with open('npors_train.jsonl', 'w') as f:
        for item in training_data[:4000]:  # 80%训练
            f.write(json.dumps(item) + '\n')
    
    with open('npors_val.jsonl', 'w') as f:
        for item in training_data[4000:]:  # 20%验证
            f.write(json.dumps(item) + '\n')
    
    # 4. 上传并训练
    train_id = upload_file('npors_train.jsonl')
    val_id = upload_file('npors_val.jsonl')
    
    # 5. 启动fine-tuning
    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model="gpt-4o-mini",
        hyperparameters={"n_epochs": 3}
    )
    
    print(f"Fine-tuning started! Job ID: {job.id}")
    return job.id

# 执行
job_id = quick_start_finetuning()
```

## 总结

本方案基于现有的NPORS 2024数据结构和已实现的提示策略，通过fine-tuning进一步提升预测准确率。建议先进行小规模试点验证，成功后再进行全量实施。混合使用fine-tuned模型和现有few-shot方法可达到最佳效果。