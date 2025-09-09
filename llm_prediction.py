#!/usr/bin/env python3
"""
LLM Prediction module for survey response prediction

This module contains the core functionality for predicting survey responses using LLMs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config_manager import ConfigManager

class LLMPredictor:
    """LLM-based survey response predictor."""
    
    def __init__(self, use_azure: bool = True):
        self.config_manager = ConfigManager()
        self.use_azure = use_azure
        self.client, self.config = self._initialize_client()
        
        # Define mapping dictionaries
        self.race_labels = {
            1: "White", 2: "Black", 3: "Asian",
            4: "American Indian or Alaska Native",
            5: "Native Hawaiian or Other Pacific Islander", 6: "Other"
        }
        
        self.birthplace_map = {
            1: "50 U.S. states or D.C.", 2: "Puerto Rico",
            3: "U.S. territory", 4: "Another country other than U.S."
        }
        
        self.gender_map = {1: "Man", 2: "Woman", 3: "Some other way"}
        
        self.marital_map = {
            1: "Married", 2: "Living with a partner", 3: "Divorced",
            4: "Separated", 5: "Widowed", 6: "Never married"
        }
        
        self.education_map = {
            1: "No school", 2: "Kindergarten to grade 11", 3: "High school graduate",
            4: "Some college, no degree", 5: "Associate degree",
            6: "Bachelor's degree", 7: "Master's degree or higher"
        }
        
        self.division_map = {
            1: "New England (CT, ME, MA, NH, RI, VT)", 2: "Middle Atlantic (NJ, NY, PA)",
            3: "East North Central (IL, IN, MI, OH, WI)", 4: "West North Central (IA, KS, MN, MO, NE, ND, SD)",
            5: "South Atlantic (DE, DC, FL, GA, MD, NC, SC, VA, WV)", 6: "East South Central (AL, KY, MS, TN)",
            7: "West South Central (AR, LA, OK, TX)", 8: "Mountain (AZ, CO, ID, MT, NV, NM, UT, WY)",
            9: "Pacific (AK, CA, HI, OR, WA)"
        }
        
        self.income_map = {
            1: "<$30,000", 2: "$30,000–39,999", 3: "$40,000–49,999", 4: "$50,000–59,999",
            5: "$60,000–69,999", 6: "$70,000–79,999", 7: "$80,000–89,999",
            8: "$90,000–99,999", 9: "$100,000+"
        }
        
        self.region_map = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
        self.metro_map = {1: "Non-metropolitan", 2: "Metropolitan"}
        
        # Default questions
        self.questions = {
            "ECON1MOD": "How would you rate the economic conditions in your community today? Answer on a scale of 1 to 4, where 1 is excellent, 2 is good, 3 is only fair, and 4 is poor."
        }
    
    def _initialize_client(self):
        """Initialize the appropriate client based on configuration."""
        if self.use_azure:
            return self.config_manager.get_azure_client()
        else:
            return None, None  # Simplified for now
    
    def build_demographic_prompt(self, row: pd.Series) -> str:
        """Build demographic prompt for a survey respondent."""
        prompt = (
            f"You are a respondent in a survey at the time of May 1st, 2024. "
            f"You are a {row['AGE']}-year-old {self.gender_map.get(row['GENDER'], 'Unknown')} "
            f"who is {row.get('RACE_TEXT', 'Unknown')}. "
            f"You were born in {self.birthplace_map.get(row['BIRTHPLACE'], 'Unknown')}, "
            f"and are currently {self.marital_map.get(row['MARITAL'], 'Unknown')}, "
            f"and have an education level of {self.education_map.get(row['EDUCATION'], 'Unknown')}. "
            f"Your annual household income is {self.income_map.get(row['INC_SDT1'], 'Unknown')}. "
            f"You live in the {self.division_map.get(row['DIVISION'], 'Unknown')} region "
            f"and are located in a {self.metro_map.get(row['METRO'], 'Unknown')} area.\\n"
            f"For each question, pay attention to the instruction especially the output format. "
            f"Think step by step. Answer in a concise way.\\n\\n"
        )
        return prompt
    
    def predict_basic(self, df: pd.DataFrame, questions: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Basic prompting prediction."""
        if questions is None:
            questions = self.questions
        
        # Initialize LLM response columns
        for q in questions.keys():
            df[q + '_LLM'] = None
        
        # Run LLM inference
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            base_prompt = self.build_demographic_prompt(row)
            
            question_block = ""
            for qid, qtext in questions.items():
                question_block += f"Question ({qid}): {qtext}\\nPlease output the number only.\\n"
            
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": base_prompt},
                        {"role": "user", "content": question_block}
                    ],
                    max_tokens=800,
                    temperature=1.0,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    model=self.config["azure_openai_deployment_name"]
                )
                
                # Parse responses
                raw = response.choices[0].message.content.strip().splitlines()
                responses = [int(line.strip().split()[-1]) for line in raw if line.strip() and line.strip()[-1].isdigit()]
                
                for i, qid in enumerate(questions):
                    if i < len(responses):
                        df.at[idx, qid + '_LLM'] = responses[i]
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                for qid in questions:
                    df.at[idx, qid + '_LLM'] = None
        
        return df

def main():
    """CLI for LLM prediction."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_prediction.py <data_file> [method] [output_file]")
        print("Methods: basic")
        return
    
    data_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "basic"
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"{data_file.replace('.csv', '')}_{method}_prediction.csv"
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Initialize predictor
    predictor = LLMPredictor(use_azure=True)
    
    # Run prediction
    if method == "basic":
        df = predictor.predict_basic(df)
    else:
        print(f"Unknown method: {method}")
        return
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()