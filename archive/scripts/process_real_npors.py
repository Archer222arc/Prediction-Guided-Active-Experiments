#!/usr/bin/env python3
"""
Process real NPORS data if available and create fine-tuning dataset
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys
from prepare_finetuning_data import (
    race_labels, gender_map, birthplace_map, marital_map, 
    education_map, division_map, income_map, metro_map,
    questions, build_demographic_prompt, create_training_sample,
    validate_jsonl_format
)

def process_real_npors_data(file_path: str, output_dir: str = "finetuning_data_real"):
    """Process real NPORS data for fine-tuning"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading real NPORS data from: {file_path}")
    
    try:
        # Try different file formats
        if file_path.endswith('.sav'):
            import pyreadstat
            df, meta = pyreadstat.read_sav(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        print(f"Loaded {len(df)} samples from real data")
        
        # Check if RACE_TEXT exists, if not create it
        if 'RACE_TEXT' not in df.columns:
            print("Creating RACE_TEXT column...")
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
        
        # Filter valid responses
        valid_mask = (
            (df['ECON1MOD'].notna()) & 
            (df['ECON1MOD'].between(1, 4)) &
            (df['AGE'].notna()) &
            (df['GENDER'].notna()) &
            (df['EDUCATION'].notna()) &
            (df['RACE_TEXT'].notna())
        )
        
        df_clean = df[valid_mask].copy()
        print(f"Valid samples after filtering: {len(df_clean)}")
        
        if len(df_clean) < 100:
            print("Warning: Very few valid samples. Check data quality.")
            return None, None
        
        # Create training samples
        training_samples = []
        
        for idx, row in df_clean.iterrows():
            for question_id, question_text in questions.items():
                if pd.notna(row[question_id]) and row[question_id] in [1, 2, 3, 4]:
                    sample = create_training_sample(row, question_id, question_text)
                    training_samples.append(sample)
        
        print(f"Generated {len(training_samples)} training samples")
        
        if len(training_samples) < 100:
            print("Error: Not enough valid training samples generated")
            return None, None
        
        # Split into train/validation (80/20)
        import random
        random.seed(42)
        random.shuffle(training_samples)
        split_idx = int(0.8 * len(training_samples))
        
        train_samples = training_samples[:split_idx]
        val_samples = training_samples[split_idx:]
        
        print(f"Train samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        
        # Save as JSONL files
        train_file = output_path / "train_real.jsonl"
        val_file = output_path / "validation_real.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Real training data saved to: {train_file}")
        print(f"Real validation data saved to: {val_file}")
        
        # Create sample preview
        preview_file = output_path / "sample_preview_real.json"
        with open(preview_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples[:5], f, indent=2, ensure_ascii=False)
        
        print(f"Sample preview saved to: {preview_file}")
        
        # Create data statistics
        stats_file = output_path / "data_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("=== NPORS Real Data Statistics ===\n\n")
            f.write(f"Total original samples: {len(df)}\n")
            f.write(f"Valid samples after filtering: {len(df_clean)}\n")
            f.write(f"Training samples generated: {len(training_samples)}\n")
            f.write(f"Train/Validation split: {len(train_samples)}/{len(val_samples)}\n\n")
            
            f.write("Response distribution for ECON1MOD:\n")
            for i in range(1, 5):
                count = (df_clean['ECON1MOD'] == i).sum()
                pct = count / len(df_clean) * 100
                f.write(f"  {i}: {count} ({pct:.1f}%)\n")
            
            f.write("\nDemographic breakdown:\n")
            f.write(f"Age range: {df_clean['AGE'].min()}-{df_clean['AGE'].max()}\n")
            f.write(f"Gender distribution:\n")
            for gender, count in df_clean['GENDER'].value_counts().items():
                pct = count / len(df_clean) * 100
                gender_label = gender_map.get(gender, f"Unknown({gender})")
                f.write(f"  {gender_label}: {count} ({pct:.1f}%)\n")
        
        print(f"Data statistics saved to: {stats_file}")
        
        return train_file, val_file
        
    except Exception as e:
        print(f"Error processing real data: {e}")
        return None, None

def main():
    """Main function to process real NPORS data"""
    
    # Look for possible NPORS data files
    possible_files = [
        "NPORS_2024_for_public_release.sav",
        "NPORS_2024_for_public_release_updated.csv",
        "NPORS_2024_for_public_release_with_LLM_prediction.csv",
        "data/NPORS_2024_for_public_release.sav",
        "data/NPORS_2024_for_public_release_updated.csv"
    ]
    
    found_file = None
    for file_path in possible_files:
        if Path(file_path).exists():
            found_file = file_path
            break
    
    if found_file:
        print(f"Found real NPORS data file: {found_file}")
        train_file, val_file = process_real_npors_data(found_file)
        
        if train_file and val_file:
            # Validate format
            print("\n=== Validating JSONL Format ===")
            train_valid = validate_jsonl_format(train_file)
            val_valid = validate_jsonl_format(val_file)
            
            if train_valid and val_valid:
                print("\n✓ Real data files are ready for Azure OpenAI fine-tuning!")
                print(f"\nFiles generated:")
                print(f"- Training: {train_file}")
                print(f"- Validation: {val_file}")
            else:
                print("\n✗ Format validation failed.")
        else:
            print("Failed to process real data.")
    else:
        print("No real NPORS data files found in expected locations.")
        print("Expected files:")
        for file_path in possible_files:
            print(f"  - {file_path}")
        print("\nPlease ensure the real NPORS data file is available.")

if __name__ == "__main__":
    main()