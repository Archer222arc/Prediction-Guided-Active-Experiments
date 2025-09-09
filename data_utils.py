#!/usr/bin/env python3
"""
Data utilities for survey data processing and analysis

This module contains utility functions for processing survey data.
"""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path
from typing import Dict, List, Optional, Any

def load_sav_data(file_path: str) -> tuple[pd.DataFrame, Any]:
    """Load SPSS .sav file."""
    try:
        df, meta = pyreadstat.read_sav(file_path)
        return df, meta
    except Exception as e:
        print(f"Error loading .sav file: {e}")
        return None, None

def create_race_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create race text field from RACEMOD columns."""
    race_labels = {
        1: "White", 2: "Black", 3: "Asian",
        4: "American Indian or Alaska Native",
        5: "Native Hawaiian or Other Pacific Islander", 6: "Other"
    }
    
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
    return df

def get_question_definitions() -> Dict[str, str]:
    """Get standard question definitions."""
    return {
        "ECON1MOD": "How would you rate the economic conditions in your community today? Answer on a scale of 1 to 4, where 1 is excellent, 2 is good, 3 is only fair, and 4 is poor.",
        "UNITY": "Which statement comes closer to your own view, even if neither is exactly right? 1. Americans are united when it comes to the most important values. 2. Americans are divided when it comes to the most important values.",
        "GPT1": "How much, if anything, have you heard about ChatGPT, an artificial intelligence (AI) program used to create text? 1. A lot, 2. A little, 3. Nothing at all.",
        "MOREGUNIMPACT": "If more Americans owned guns, do you think there would be... 1. More crime, 2. Less crime, 3. No difference.",
        "GAMBLERESTR": "Which statement comes closest to your views about gambling where you live? 1.There should be MORE restrictions on gambling than there are today. 2. Restrictions on gambling are about right. 3.There should be FEWER restrictions on gambling than there are today."
    }

def calculate_prediction_accuracy(df: pd.DataFrame, true_col: str, pred_col: str) -> Dict[str, float]:
    """Calculate prediction accuracy metrics."""
    valid_mask = df[true_col].notna() & df[pred_col].notna()
    true_vals = df.loc[valid_mask, true_col]
    pred_vals = df.loc[valid_mask, pred_col]
    
    if len(true_vals) == 0:
        return {"error": "No valid predictions"}
    
    # Accuracy
    accuracy = (true_vals == pred_vals).mean()
    
    # Mean Absolute Error
    mae = np.abs(true_vals - pred_vals).mean()
    
    return {
        "accuracy": accuracy,
        "mae": mae,
        "n_predictions": len(true_vals)
    }

def prepare_data_for_analysis(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Prepare raw data for analysis."""
    # Load data
    if file_path.endswith('.sav'):
        df, meta = load_sav_data(file_path)
        if df is None:
            return None
    else:
        df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} rows from {file_path}")
    
    # Create race text if RACEMOD columns exist
    if any(f"RACEMOD_{i}" in df.columns for i in range(1, 7)):
        df = create_race_text(df)
        print("Created RACE_TEXT column")
    
    # Save processed data
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    
    return df

def main():
    """CLI for data processing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_utils.py <input_file> [output_file]")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.sav', '_processed.csv')
    
    # Process data
    df = prepare_data_for_analysis(input_file, output_file)
    
    if df is not None:
        print(f"\\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()