# LLM Prediction Analysis Documentation

## Overview

This document summarizes the data structure and LLM prediction methodology from the `LLM_prediction.ipynb` notebook, which implements various prompting strategies for predicting survey responses using Large Language Models (LLMs).

## Dataset Structure

### Source Data
- **Dataset**: NPORS (National Public Opinion Reference Survey) 2024
- **Original file**: `NPORS_2024_for_public_release.sav`
- **Format**: SPSS .sav file converted to CSV
- **Total observations**: 5,626 respondents

### Key Demographics Variables

#### 1. Race/Ethnicity
- `HISP`: Hispanic status (1=Hispanic, 0=Not Hispanic)
- `RACEMOD_1` to `RACEMOD_6`: Multiple race selections
  - 1=White, 2=Black, 3=Asian, 4=American Indian/Alaska Native
  - 5=Native Hawaiian/Pacific Islander, 6=Other
- `RACE_TEXT`: Processed text field combining Hispanic status and race

#### 2. Personal Characteristics
- `AGE`: Respondent's age
- `GENDER`: 1=Man, 2=Woman, 3=Some other way
- `BIRTHPLACE`: 1=50 US states/DC, 2=Puerto Rico, 3=US territory, 4=Another country
- `MARITAL`: 1=Married, 2=Living with partner, 3=Divorced, 4=Separated, 5=Widowed, 6=Never married

#### 3. Socioeconomic Status
- `EDUCATION`: 1=No school to 7=Master's degree or higher
- `INC_SDT1`: Household income (1=<$30K to 9=$100K+)

#### 4. Geographic Information
- `CREGION`: Census region (1=Northeast, 2=Midwest, 3=South, 4=West)
- `DIVISION`: Census division (1-9, specific state groupings)
- `METRO`: Metropolitan status (1=Non-metropolitan, 2=Metropolitan)

### Target Questions Analyzed

#### Primary Question
- `ECON1MOD`: "How would you rate the economic conditions in your community today?"
  - Scale: 1=Excellent, 2=Good, 3=Only fair, 4=Poor

#### Additional Questions (in extended analysis)
- `UNITY`: American unity/division perception
- `GPT1`: Awareness of ChatGPT
- `MOREGUNIMPACT`: Impact of more gun ownership
- `GAMBLERESTR`: Gambling restriction preferences

## LLM Prediction Methodology

### 1. Basic Prompting Approach

#### System Prompt Structure
```
You are a respondent in a survey at the time of May 1st, 2024. 
Here is some of your demographic information:
You are a {age}-year-old {gender} who is {race}. You were born in {birthplace}, 
and are currently {marital_status}, and have an education level of {education}. 
Your annual household income is {income}. You live in the {region} region 
and are located in a {metro} area.
```

#### Chain-of-Thought Process
1. **Personal Context**: "Based on your personal information, what would be your most likely job, economic status, and political stand?"
2. **Temporal Context**: "What are the economics and societal conditions like at the time of May, 2024?"
3. **Question Response**: Direct question with specific output format requirement

#### Model Configuration
- **Model**: Azure OpenAI "o4-mini" (GPT-4 variant)
- **Temperature**: 1.0 (for probabilistic responses)
- **Max tokens**: 4,000
- **Output format**: Single integer (1-4) or probability distribution

### 2. Chain-of-Thought (CoT) Prompting

#### Enhanced Reasoning Structure
For each question type, specific reasoning aspects are provided:

**For Economic Questions (ECON1MOD)**:
1. Education and Income – Employment stability and mobility access
2. Region and Area – Local economic conditions
3. Marital Status and Age – Household financial stability
4. Race and Ethnicity – Perceived opportunity disparities

**Implementation**:
- Temperature: 0.6 (more focused responses)
- Max tokens: 2 (constrained output)
- Explicit reasoning prompts before final answer

### 3. Few-Shot Prompting

#### Example Selection Strategy
- **Grouping variables**: Income level, Education level, Division
- **Sample size**: 3 examples per demographic group
- **Example format**: Input demographics + historical responses

#### Sampling Process
```python
# Group similar demographic profiles
sample_feature = ['INCOME_LEVEL', 'EDUCATION_LEVEL', 'DIVISION']
# Provide 3 examples per group
sampled_df = df.groupby(sample_feature).apply(
    lambda x: x.sample(n=3) if len(x) >= 3 else x
)
```

## Data Processing Pipeline

### 1. Data Preparation
```python
# Feature selection
features = ['AGE', 'GENDER', 'RACE_TEXT', 'BIRTHPLACE', 
           'MARITAL', 'EDUCATION', 'DIVISION', 'INC_SDT1', 'METRO']

# Race text processing
race_texts = []
for row in df.iterrows():
    races = [race_labels[i] for i in range(1,7) if row[f"RACEMOD_{i}"] == 1]
    race_text = "Hispanic" if row["HISP"] == 1 else "Not Hispanic"
    if races: race_text += ", " + ", ".join(races)
    race_texts.append(race_text)
```

### 2. LLM Inference Loop
```python
for idx, row in df.iterrows():
    # Build demographic prompt
    base_prompt = construct_demographic_prompt(row)
    
    # For each question
    for qid, qtext in questions.items():
        # Apply prompting strategy (basic/CoT/few-shot)
        response = client.chat.completions.create(...)
        df.at[idx, qid + '_LLM'] = parse_response(response)
```

### 3. Response Processing
- **Probabilistic outputs**: Parse comma-separated probabilities
- **Direct outputs**: Extract single integer responses
- **Error handling**: Track failed responses for retry

## PGAE Implementation

### Prediction-Guided Active Experiments

#### Core Algorithm
```python
def get_PGAE_design(df, X, F, Y, gamma):
    # Compute conditional statistics
    summary = df.groupby(X + [F])[Y].agg(
        cond_mean='mean', cond_var='var'
    )
    
    # Calculate sampling probabilities
    group_stats['exp_prob'] = sqrt(mean_of_cond_var / var_of_cond_mean)
    group_stats['accept_prob'] = sqrt(var_of_cond_mean) / max(sqrt(var_of_cond_var))
    
    return group_stats
```

#### Adaptive Design
- **Batch size**: 100 samples
- **Target labels**: 500 true labels
- **Budget parameter**: γ = 0.5
- **Update frequency**: After each batch

#### Performance Metrics
- **PGAE Results**: MSE=0.0013, Coverage=99.9%, CI length=0.1394
- **Adaptive PGAE**: MSE=0.0032, Coverage=64.2%, CI length=0.1180  
- **Active Learning**: MSE=0.0015, Coverage=95.9%, CI length=0.1572

## Key Findings

### 1. Prediction Accuracy by Method
- **Few-shot prompting** showed best performance for demographic-based prediction
- **Chain-of-thought** improved reasoning quality but required careful prompt design
- **Basic prompting** provided baseline performance with simpler implementation

### 2. Demographic Factors
- **Education level** strongly correlated with prediction accuracy
- **Geographic region** influenced both actual and predicted responses
- **Income level** showed interaction effects with other demographics

### 3. Response Distribution Analysis
```python
# True vs Predicted proportions for ECON1MOD
Choice 1: true=5.1%, LLM=6.8%
Choice 2: true=36.8%, LLM=41.5%  
Choice 3: true=38.2%, LLM=36.4%
Choice 4: true=19.7%, LLM=15.4%
```

## Technical Configuration

### Azure OpenAI Setup
```python
endpoint = "https://chenh-m9vmscvr-eastus2.cognitiveservices.azure.com/"
model_name = "o4-mini"
deployment = "o4-mini"
api_version = "2024-12-01-preview"
```

### Parallel Processing
- **ThreadPoolExecutor**: Up to 10 concurrent workers
- **Error handling**: Automatic retry for rate limits
- **Progress tracking**: tqdm progress bars for long-running processes

## File Outputs

### Generated Datasets

#### 1. `NPORS_2024_for_public_release_updated.csv`
- **内容**: 原始NPORS数据 + 处理后的RACE_TEXT列
- **用途**: 基础数据文件，包含人口统计信息和真实调查回答
- **特点**: 纯净的人类响应数据，适合fine-tuning使用

#### 2. `NPORS_2024_for_public_release_with_LLM_prediction.csv`
- **生成逻辑**: 在updated.csv基础上添加LLM预测列
- **代码过程**:
  ```python
  # 为每个问题初始化LLM预测列
  for q in questions.keys():
      df[q + '_LLM'] = None          # 最终预测值
      df[q + '_LLM_raw'] = None      # 原始模型输出
  
  # 对每个受访者进行推理
  for idx, row in df.iterrows():
      for qid, qtext in questions.items():
          # 构建人口统计提示
          # 调用Azure OpenAI API
          # 解析和保存预测结果
          df.at[idx, qid + '_LLM'] = predicted_response
  
  df.to_csv("with_LLM_prediction.csv", index=False)
  ```
- **新增列结构**:
  - `ECON1MOD_LLM`: 经济状况评估的LLM预测
  - `UNITY_LLM`: 美国团结程度的LLM预测  
  - `GPT1_LLM`: ChatGPT认知度的LLM预测
  - `MOREGUNIMPACT_LLM`: 枪支影响的LLM预测
  - `GAMBLERESTR_LLM`: 赌博限制的LLM预测
  - `{QUESTION}_LLM_raw`: 每个问题的原始模型文本输出

#### 3. 中间结果文件
- `NPORS_2024_for_public_release_basic_prompting.csv` - Basic prompting结果
- `NPORS_2024_for_public_release_cot_prompting_tem_0.6.csv` - CoT结果  
- `NPORS_2024_for_public_release_few_shot_prompting.csv` - Few-shot结果

### 文件区别总结

| 特征 | updated.csv | with_LLM_prediction.csv |
|------|-------------|-------------------------|
| **数据来源** | 纯人类调查响应 | 人类响应 + LLM预测 |
| **列数量** | ~80列 | ~90+列 (添加LLM预测列) |
| **用途** | Fine-tuning训练数据 | 性能评估、PGAE研究 |
| **数据纯净性** | ✅ 纯净 | ❌ 包含模型输出 |
| **Fine-tuning适用性** | ✅ 推荐 | ❌ 不推荐 (循环训练问题) |
| **研究价值** | 基础数据 | 预测性能分析 |

### 重要说明

**Fine-tuning数据选择**:
- ✅ **使用**: `updated.csv` - 确保模型学习真实人类响应模式
- ❌ **避免**: `with_LLM_prediction.csv` - 防止模型学习自己的预测偏差

**LLM预测文件的正确用途**:
1. **性能基准**: 评估不同提示策略的效果
2. **错误分析**: 识别模型在特定人口群体上的弱点  
3. **PGAE框架**: 在主动学习中使用预测来指导采样策略
4. **研究对比**: 分析模型预测与人类实际回答的差异

### Performance Tracking
- **Error indices**: Tracked failed API calls
- **Confidence intervals**: 90% confidence level (α=0.90)
- **Replication**: 1,000 bootstrap replications for stability analysis