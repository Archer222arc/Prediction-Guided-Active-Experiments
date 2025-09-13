#!/usr/bin/env python3
"""
Prepare fine-tuning data for Azure OpenAI based on NPORS 2024 dataset
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Any
from collections import defaultdict

# Mapping dictionaries from the notebook
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
    1: "New England (CT, ME, MA, NH, RI, VT)",
    2: "Middle Atlantic (NJ, NY, PA)",
    3: "East North Central (IL, IN, MI, OH, WI)",   
    4: "West North Central (IA, KS, MN, MO, NE, ND, SD)",
    5: "South Atlantic (DE, DC, FL, GA, MD, NC, SC, VA, WV)",
    6: "East South Central (AL, KY, MS, TN)",
    7: "West South Central (AR, LA, OK, TX)",
    8: "Mountain (AZ, CO, ID, MT, NV, NM, UT, WY)",
    9: "Pacific (AK, CA, HI, OR, WA)"
}

income_map = {
    1: "<$30,000", 2: "$30,000–39,999", 
    3: "$40,000–49,999", 4: "$50,000–59,999",
    5: "$60,000–69,999", 6: "$70,000–79,999", 
    7: "$80,000–89,999", 8: "$90,000–99,999", 
    9: "$100,000+"
}

metro_map = {1: "Non-metropolitan", 2: "Metropolitan"}

region_map = {
    1: "Northeast", 2: "Midwest", 
    3: "South", 4: "West"
}

# Complete set of 5 survey questions
questions = {
    "ECON1MOD": "How would you rate the economic conditions in your community today? Answer on a scale of 1 to 4, where 1 is excellent, 2 is good, 3 is only fair, and 4 is poor.",
    "UNITY": "Which statement comes closer to your own view, even if neither is exactly right? 1. Americans are united when it comes to the most important values. 2. Americans are divided when it comes to the most important values.",
    "GPT1": "How much, if anything, have you heard about ChatGPT, an artificial intelligence (AI) program used to create text? 1. A lot, 2. A little, 3. Nothing at all.",
    "MOREGUNIMPACT": "If more Americans owned guns, do you think there would be... 1. More crime, 2. Less crime, 3. No difference.",
    "GAMBLERESTR": "Which statement comes closest to your views about gambling where you live? 1. There should be MORE restrictions on gambling than there are today. 2. Restrictions on gambling are about right. 3. There should be FEWER restrictions on gambling than there are today."
}

# Response ranges for each question
response_ranges = {
    "ECON1MOD": [1, 2, 3, 4],
    "UNITY": [1, 2],
    "GPT1": [1, 2, 3],
    "MOREGUNIMPACT": [1, 2, 3],
    "GAMBLERESTR": [1, 2, 3]
}

def create_synthetic_npors_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Create synthetic NPORS-like data for demonstration
    In real implementation, replace this with actual data loading
    """
    np.random.seed(42)
    
    data = {
        'AGE': np.random.randint(18, 85, n_samples),
        'GENDER': np.random.choice([1, 2, 3], n_samples, p=[0.48, 0.49, 0.03]),
        'HISP': np.random.choice([0, 1], n_samples, p=[0.82, 0.18]),
        'BIRTHPLACE': np.random.choice([1, 2, 3, 4], n_samples, p=[0.86, 0.02, 0.01, 0.11]),
        'MARITAL': np.random.choice(list(range(1, 7)), n_samples),
        'EDUCATION': np.random.choice(list(range(1, 8)), n_samples),
        'DIVISION': np.random.choice(list(range(1, 10)), n_samples),
        'INC_SDT1': np.random.choice(list(range(1, 10)), n_samples),
        'METRO': np.random.choice([1, 2], n_samples, p=[0.2, 0.8]),
        # Generate responses for all 5 questions
        'ECON1MOD': np.random.choice([1, 2, 3, 4], n_samples, p=[0.05, 0.37, 0.38, 0.20]),
        'UNITY': np.random.choice([1, 2], n_samples, p=[0.4, 0.6]),
        'GPT1': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3]),
        'MOREGUNIMPACT': np.random.choice([1, 2, 3], n_samples, p=[0.45, 0.25, 0.30]),
        'GAMBLERESTR': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create race columns (simplified)
    for i in range(1, 7):
        df[f'RACEMOD_{i}'] = 0
    
    # Assign primary race
    race_choices = np.random.choice(list(range(1, 7)), n_samples, p=[0.7, 0.13, 0.06, 0.02, 0.01, 0.08])
    for idx, race in enumerate(race_choices):
        df.loc[idx, f'RACEMOD_{race}'] = 1
    
    # Create RACE_TEXT
    race_texts = []
    for idx, row in df.iterrows():
        races = []
        for i in range(1, 7):
            if row.get(f"RACEMOD_{i}", 0) == 1:
                races.append(race_labels[i])
        race_text = "Hispanic" if row["HISP"] == 1 else "Not Hispanic"
        if races:
            race_text += ", " + ", ".join(races)
        race_texts.append(race_text)
    
    df['RACE_TEXT'] = race_texts
    return df

def build_demographic_system_prompt(row: pd.Series) -> str:
    """Build optimized demographic system prompt"""
    return (
        f"You are a respondent in a survey at the time of May 1st, 2024. "
        f"You are a {row['AGE']}-year-old {gender_map.get(row['GENDER'], 'Unknown')} "
        f"who is {row.get('RACE_TEXT', 'Unknown')}. "
        f"You were born in {birthplace_map.get(row['BIRTHPLACE'], 'Unknown')}, "
        f"and are currently {marital_map.get(row['MARITAL'], 'Unknown')}. "
        f"You have an education level of {education_map.get(row['EDUCATION'], 'Unknown')}. "
        f"Your annual household income is {income_map.get(row['INC_SDT1'], 'Unknown')}. "
        f"You live in the {division_map.get(row['DIVISION'], 'Unknown')} region "
        f"and are located in a {metro_map.get(row['METRO'], 'Unknown')} area. "
        f"Answer survey questions based on your demographic profile and personal circumstances. "
        f"Be realistic and consistent with your background."
    )

def create_training_sample(row: pd.Series, question_id: str, question_text: str) -> Dict[str, Any]:
    """Create optimized training sample with improved prompt structure"""
    
    # Ensure response is valid
    response = row[question_id]
    if pd.isna(response) or response not in response_ranges[question_id]:
        return None
    
    system_message = build_demographic_system_prompt(row)
    
    user_message = f"Question: {question_text}\nPlease output the number only."
    
    assistant_message = str(int(response))
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }

def check_data_balance(samples: List[Dict]) -> Dict[str, Dict]:
    """Check training data balance across questions and responses"""
    
    question_stats = defaultdict(lambda: defaultdict(int))
    
    for sample in samples:
        user_msg = sample['messages'][1]['content']
        response = sample['messages'][2]['content']
        
        # Identify question type
        question_found = None
        question_key_phrases = {
            "ECON1MOD": "economic conditions",
            "UNITY": "Americans are united",
            "GPT1": "ChatGPT", 
            "MOREGUNIMPACT": "more Americans owned guns",
            "GAMBLERESTR": "gambling"
        }
        
        for qid, key_phrase in question_key_phrases.items():
            if key_phrase.lower() in user_msg.lower():
                question_found = qid
                break
        
        if question_found:
            question_stats[question_found][response] += 1
    
    # Print statistics
    print("\n=== 数据平衡性分析 ===")
    total_samples = len(samples)
    
    for qid in questions.keys():
        stats = question_stats[qid]
        total_q = sum(stats.values())
        print(f"\n{qid} (总计: {total_q} 样本, {total_q/total_samples*100:.1f}%):")
        
        valid_responses = response_ranges[qid]
        for value in valid_responses:
            count = stats.get(str(value), 0)
            if total_q > 0:
                print(f"  选项 {value}: {count} ({count/total_q*100:.1f}%)")
            else:
                print(f"  选项 {value}: {count} (0%)")
    
    return dict(question_stats)

def create_training_data(df: pd.DataFrame, output_dir: str = "finetuning_data"):
    """Convert NPORS data to Azure fine-tuning format"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Processing {len(df)} samples...")
    
    # Filter valid responses for all questions
    required_cols = ['AGE', 'GENDER', 'EDUCATION', 'RACE_TEXT'] + list(questions.keys())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 缺少列 {missing_cols}")
    
    valid_mask = df[required_cols].notna().all(axis=1)
    
    # Additional validation for response ranges (exclude 99.0 = "don't know/refuse")
    for qid in questions.keys():
        if qid in df.columns:
            valid_responses = response_ranges[qid]
            # Exclude 99.0 (don't know/refuse to answer) and NaN values
            valid_mask &= (df[qid].isin(valid_responses)) & (df[qid] != 99.0)
    
    df_clean = df[valid_mask].copy()
    print(f"过滤后有效样本: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("错误: 没有有效样本")
        return None, None
    
    # Create training samples
    training_samples = []
    
    for idx, row in df_clean.iterrows():
        for question_id, question_text in questions.items():
            if question_id in row and pd.notna(row[question_id]):
                sample = create_training_sample(row, question_id, question_text)
                if sample:  # Only add if sample is valid
                    training_samples.append(sample)
    
    print(f"生成训练样本: {len(training_samples)}")
    
    if len(training_samples) < 100:
        print("警告: 训练样本数量较少，可能影响fine-tuning效果")
    
    # Check data balance
    balance_stats = check_data_balance(training_samples)
    
    # Split into train/validation (80/20)
    random.seed(42)
    random.shuffle(training_samples)
    split_idx = int(0.8 * len(training_samples))
    
    train_samples = training_samples[:split_idx]
    val_samples = training_samples[split_idx:]
    
    print(f"\n=== 数据分割 ===")
    print(f"训练样本: {len(train_samples)}")
    print(f"验证样本: {len(val_samples)}")
    
    # Save as JSONL files
    train_file = output_path / "train.jsonl"
    val_file = output_path / "validation.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")
    
    # Create sample preview
    preview_file = output_path / "sample_preview.json"
    with open(preview_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples[:5], f, indent=2, ensure_ascii=False)
    
    print(f"Sample preview saved to: {preview_file}")
    
    return train_file, val_file

def validate_jsonl_format(file_path: Path) -> tuple:
    """Enhanced JSONL file validation"""
    
    valid_count = 0
    error_count = 0
    question_counts = defaultdict(int)
    
    print(f"\n=== 验证文件: {file_path} ===")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    
                    # Check required structure
                    assert 'messages' in data, "Missing 'messages' field"
                    assert len(data['messages']) == 3, f"Expected 3 messages, got {len(data['messages'])}"
                    assert data['messages'][0]['role'] == 'system', "First message should be system"
                    assert data['messages'][1]['role'] == 'user', "Second message should be user"
                    assert data['messages'][2]['role'] == 'assistant', "Third message should be assistant"
                    
                    # Check content
                    system_content = data['messages'][0]['content']
                    user_content = data['messages'][1]['content']
                    assistant_content = data['messages'][2]['content']
                    
                    assert len(system_content) > 0, "Empty system content"
                    assert len(user_content) > 0, "Empty user content"
                    assert len(assistant_content) > 0, "Empty assistant content"
                    
                    # Validate response format
                    response = assistant_content.strip()
                    assert response.isdigit(), f"Invalid response format: {response}"
                    response_int = int(response)
                    assert 1 <= response_int <= 4, f"Response out of range: {response_int}"
                    
                    # Count question types
                    question_key_phrases = {
                        "ECON1MOD": "economic conditions",
                        "UNITY": "Americans are united",
                        "GPT1": "ChatGPT",
                        "MOREGUNIMPACT": "more Americans owned guns",
                        "GAMBLERESTR": "gambling"
                    }
                    
                    for qid, key_phrase in question_key_phrases.items():
                        if key_phrase.lower() in user_content.lower():
                            question_counts[qid] += 1
                            break
                    
                    valid_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only show first 5 errors
                        print(f"第{i}行错误: {e}")
        
        print(f"\n验证完成:")
        print(f"  有效样本: {valid_count}")
        print(f"  错误样本: {error_count}")
        print(f"  成功率: {valid_count/(valid_count+error_count)*100:.1f}%")
        
        print(f"\n问题类型分布:")
        for qid, count in question_counts.items():
            print(f"  {qid}: {count} 样本")
        
        return valid_count, error_count, question_counts
        
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return 0, 1, {}

def main():
    """Main execution function"""
    print("=== Azure OpenAI Fine-tuning Data Preparation ===\n")
    
    # Check for existing data files (prioritize real data)
    data_files = [
        "data/NPORS_2024_for_public_release_updated.csv",
        "data/NPORS_2024_for_public_release.sav", 
        "NPORS_2024_for_public_release_updated.csv",
        "NPORS_2024_for_public_release_with_LLM_prediction.csv"
    ]
    
    df = None
    for file_name in data_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"发现真实数据文件: {file_name}")
            try:
                if file_name.endswith('.sav'):
                    try:
                        import pyreadstat
                        df, meta = pyreadstat.read_sav(file_path)
                        print(f"成功加载 {len(df)} 个样本从 .sav 文件: {file_name}")
                    except ImportError:
                        print("需要安装 pyreadstat 来读取 .sav 文件: pip install pyreadstat")
                        continue
                else:
                    df = pd.read_csv(file_path)
                    print(f"成功加载 {len(df)} 个样本从 .csv 文件: {file_name}")
                # Check if RACE_TEXT exists, if not create it
                if 'RACE_TEXT' not in df.columns:
                    print("创建 RACE_TEXT 列...")
                    race_texts = []
                    for idx, row in df.iterrows():
                        races = []
                        for i in range(1, 7):
                            if row.get(f"RACEMOD_{i}", 0) == 1:
                                races.append(race_labels[i])
                        race_text = "Hispanic" if row.get("HISP", 0) == 1 else "Not Hispanic"
                        if races:
                            race_text += ", " + ", ".join(races)
                        race_texts.append(race_text)
                    df['RACE_TEXT'] = race_texts
                    print("RACE_TEXT 列创建完成")
                
                break
            except Exception as e:
                print(f"加载 {file_name} 时出错: {e}")
    
    # If no existing data found, create synthetic data
    if df is None:
        print("No existing NPORS data found. Creating synthetic dataset for demonstration...")
        df = create_synthetic_npors_data(n_samples=5000)
        
        # Save synthetic data
        synthetic_file = "synthetic_npors_data.csv"
        df.to_csv(synthetic_file, index=False)
        print(f"Synthetic data saved to: {synthetic_file}")
    
    # Create training data
    train_file, val_file = create_training_data(df)
    
    if train_file and val_file:
        # Validate format
        print("\n=== 验证JSONL格式 ===")
        train_valid, train_errors, train_q_counts = validate_jsonl_format(train_file)
        val_valid, val_errors, val_q_counts = validate_jsonl_format(val_file)
        
        if train_errors == 0 and val_errors == 0:
            print("\n✅ 所有文件已准备就绪，可用于Azure OpenAI fine-tuning!")
            print(f"\n下一步操作:")
            print(f"1. 上传 {train_file} 作为训练数据")
            print(f"2. 上传 {val_file} 作为验证数据")
            print(f"3. 在Azure OpenAI Studio中配置fine-tuning作业")
            print(f"\n📊 推荐的fine-tuning参数:")
            print(f"- Learning rate: 1e-5 到 5e-5")
            print(f"- Batch size: 8 到 16")
            print(f"- Epochs: 3 到 5")
        else:
            print("\n❌ 格式验证失败，请检查上述错误。")
    else:
        print("❌ 数据处理失败。")

if __name__ == "__main__":
    main()