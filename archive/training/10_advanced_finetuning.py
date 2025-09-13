#!/usr/bin/env python3
"""
高级微调脚本 - 解决系统性偏差，优化超参数
针对发现的预测过度集中问题进行全面优化
"""

import os
import time
import json
import pandas as pd
import numpy as np
from collections import Counter
from openai import OpenAI
import logging
from typing import Dict, List, Tuple
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFineTuner:
    """高级微调器 - 专门处理数据平衡和超参数优化"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        azure_config = config["azure_endpoints"]["north_central_us"]
        
        self.client = OpenAI(
            api_key=azure_config["api_key"],
            base_url=azure_config["endpoint"].rstrip("/") + "/openai/v1/"
        )
        
        # 问题配置
        self.questions = {
            'ECON1MOD': {'valid_options': [1, 2, 3, 4], 'weight': 1.0},
            'UNITY': {'valid_options': [1, 2], 'weight': 2.0},  # 增加权重解决过度集中
            'GPT1': {'valid_options': [1, 2, 3], 'weight': 1.5},
            'MOREGUNIMPACT': {'valid_options': [1, 2, 3], 'weight': 1.0},
            'GAMBLERESTR': {'valid_options': [1, 2, 3, 4, 5], 'weight': 2.0}  # 增加权重解决过度集中
        }
        
        # 多套超参数配置用于实验
        self.hyperparameter_configs = {
            'conservative': {
                'n_epochs': 2,
                'batch_size': 16,
                'learning_rate_multiplier': 0.05,
                'description': '保守配置 - 防止过拟合'
            },
            'moderate': {
                'n_epochs': 3, 
                'batch_size': 32,
                'learning_rate_multiplier': 0.1,
                'description': '适中配置 - 平衡性能'
            },
            'aggressive': {
                'n_epochs': 4,
                'batch_size': 64,
                'learning_rate_multiplier': 0.2,
                'description': '激进配置 - 追求最佳性能'
            },
            'balanced_focus': {
                'n_epochs': 3,
                'batch_size': 24,
                'learning_rate_multiplier': 0.15,
                'description': '平衡重点配置 - 专门解决偏差问题'
            }
        }
        
        logger.info("高级微调器初始化完成")
    
    def analyze_data_distribution(self, data_file: str) -> Dict:
        """分析训练数据的分布情况"""
        
        logger.info("分析数据分布...")
        
        # 读取训练数据
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # 解析数据
        question_distributions = {}
        total_samples = len(lines)
        
        for question in self.questions.keys():
            distributions = {option: 0 for option in self.questions[question]['valid_options']}
            
            for line in lines:
                try:
                    data = json.loads(line)
                    # 提取目标答案
                    if 'messages' in data:
                        assistant_msg = None
                        for msg in data['messages']:
                            if msg['role'] == 'assistant':
                                assistant_msg = msg['content']
                                break
                        
                        if assistant_msg and question in assistant_msg:
                            # 简单的答案提取 (需要根据实际格式调整)
                            for option in self.questions[question]['valid_options']:
                                if f"{question}: {option}" in assistant_msg:
                                    distributions[option] += 1
                                    break
                except:
                    continue
            
            # 计算分布统计
            total_for_question = sum(distributions.values())
            if total_for_question > 0:
                percentages = {k: v/total_for_question for k, v in distributions.items()}
                
                # 计算分布的均匀程度 (熵)
                entropy = -sum(p * np.log2(p) for p in percentages.values() if p > 0)
                max_entropy = np.log2(len(self.questions[question]['valid_options']))
                balance_score = entropy / max_entropy
                
                question_distributions[question] = {
                    'counts': distributions,
                    'percentages': percentages,
                    'total_samples': total_for_question,
                    'balance_score': balance_score,
                    'is_balanced': balance_score > 0.8  # 80%平衡度阈值
                }
        
        return {
            'total_samples': total_samples,
            'question_distributions': question_distributions
        }
    
    def create_balanced_dataset(self, input_file: str, output_file: str, balance_strategy: str = 'oversample') -> str:
        """创建平衡的数据集"""
        
        logger.info(f"使用{balance_strategy}策略创建平衡数据集...")
        
        # 分析原始分布
        distribution_info = self.analyze_data_distribution(input_file)
        
        # 读取原始数据
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        balanced_lines = []
        
        if balance_strategy == 'oversample':
            # 过采样策略 - 增加少数类样本
            for line in lines:
                balanced_lines.append(line)
                
                # 根据问题类型和选项分布决定是否重复
                try:
                    data = json.loads(line)
                    if 'messages' in data:
                        # 检查是否包含需要平衡的问题
                        for question in ['UNITY', 'GAMBLERESTR']:  # 重点平衡这两个过度集中的问题
                            weight = self.questions[question]['weight']
                            if weight > 1.0 and random.random() < (weight - 1.0):
                                balanced_lines.append(line)  # 重复添加
                except:
                    continue
        
        elif balance_strategy == 'weighted':
            # 加权策略 - 修改样本权重 (通过重复实现)
            for line in lines:
                try:
                    data = json.loads(line)
                    repeat_count = 1
                    
                    # 根据内容决定重复次数
                    if 'messages' in data:
                        assistant_msg = None
                        for msg in data['messages']:
                            if msg['role'] == 'assistant':
                                assistant_msg = msg['content']
                                break
                        
                        if assistant_msg:
                            # 对UNITY选项1和GAMBLERESTR选项1,3增加权重
                            if 'UNITY: 1' in assistant_msg:
                                repeat_count = 3  # 3倍权重
                            elif 'GAMBLERESTR: 1' in assistant_msg or 'GAMBLERESTR: 3' in assistant_msg:
                                repeat_count = 2  # 2倍权重
                    
                    for _ in range(repeat_count):
                        balanced_lines.append(line)
                
                except:
                    balanced_lines.append(line)
        
        # 打乱数据
        random.shuffle(balanced_lines)
        
        # 保存平衡后的数据
        with open(output_file, 'w') as f:
            f.writelines(balanced_lines)
        
        logger.info(f"平衡数据集已保存: {output_file}")
        logger.info(f"原始样本数: {len(lines)}, 平衡后样本数: {len(balanced_lines)}")
        
        return output_file
    
    def create_finetuning_job(self, config_name: str = 'balanced_focus', 
                            use_balanced_data: bool = True,
                            balance_strategy: str = 'weighted') -> Dict:
        """创建微调作业"""
        
        logger.info(f"使用配置'{config_name}'创建微调作业")
        
        if config_name not in self.hyperparameter_configs:
            raise ValueError(f"未知配置: {config_name}")
        
        hyperparams = self.hyperparameter_configs[config_name]
        logger.info(f"配置描述: {hyperparams['description']}")
        
        # 准备数据文件
        train_file = "finetuning_data/train.jsonl"
        validation_file = "finetuning_data/validation.jsonl"
        
        if use_balanced_data:
            logger.info("创建平衡训练数据...")
            train_file = self.create_balanced_dataset(
                train_file, 
                f"finetuning_data/train_balanced_{balance_strategy}.jsonl",
                balance_strategy
            )
        
        # 上传文件
        logger.info("上传训练文件...")
        tr = self.client.files.create(
            file=open(train_file, "rb"), 
            purpose="fine-tune"
        )
        logger.info(f"训练文件ID: {tr.id}")
        
        logger.info("上传验证文件...")
        vr = self.client.files.create(
            file=open(validation_file, "rb"), 
            purpose="fine-tune"
        )
        logger.info(f"验证文件ID: {vr.id}")
        
        # 等待文件处理
        self._wait_for_file_processing(tr.id)
        self._wait_for_file_processing(vr.id)
        
        # 创建微调作业
        suffix = f"npors-advanced-{config_name}"
        if use_balanced_data:
            suffix += f"-{balance_strategy}"
        
        logger.info("创建微调作业...")
        job = self.client.fine_tuning.jobs.create(
            training_file=tr.id,
            validation_file=vr.id,
            model="gpt-4.1-mini-2025-04-14",
            suffix=suffix,
            hyperparameters={
                "n_epochs": hyperparams['n_epochs'],
                "batch_size": hyperparams['batch_size'],
                "learning_rate_multiplier": hyperparams['learning_rate_multiplier']
            }
        )
        
        # 保存作业信息
        job_info = {
            "job_id": job.id,
            "model": job.model,
            "config_name": config_name,
            "hyperparameters": hyperparams,
            "use_balanced_data": use_balanced_data,
            "balance_strategy": balance_strategy if use_balanced_data else None,
            "training_file": tr.id,
            "validation_file": vr.id,
            "created_at": job.created_at,
            "status": job.status
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_file = f"advanced_finetuning_job_{config_name}_{timestamp}.json"
        
        with open(job_file, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        logger.info(f"✅ 高级微调作业创建成功!")
        logger.info(f"作业ID: {job.id}")
        logger.info(f"配置: {config_name}")
        logger.info(f"平衡数据: {use_balanced_data}")
        logger.info(f"作业信息已保存: {job_file}")
        
        return job_info
    
    def _wait_for_file_processing(self, file_id: str, max_wait: int = 300):
        """等待文件处理完成"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            file_info = self.client.files.retrieve(file_id)
            if file_info.status == 'processed':
                return True
            elif file_info.status == 'error':
                raise Exception(f"文件处理失败: {file_id}")
            time.sleep(10)
        
        raise Exception(f"文件处理超时: {file_id}")
    
    def run_hyperparameter_sweep(self, configs: List[str] = None) -> List[Dict]:
        """运行超参数扫描"""
        
        if configs is None:
            configs = ['conservative', 'moderate', 'balanced_focus']
        
        logger.info(f"开始超参数扫描，配置: {configs}")
        
        jobs = []
        for config in configs:
            try:
                logger.info(f"\n=== 创建配置: {config} ===")
                job_info = self.create_finetuning_job(
                    config_name=config,
                    use_balanced_data=True,
                    balance_strategy='weighted'
                )
                jobs.append(job_info)
                
                # 间隔时间避免API限制
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"配置{config}创建失败: {e}")
                continue
        
        # 保存扫描结果
        sweep_info = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "configs_tested": configs,
            "jobs_created": jobs,
            "total_jobs": len(jobs)
        }
        
        with open(f"hyperparameter_sweep_{sweep_info['timestamp']}.json", 'w') as f:
            json.dump(sweep_info, f, indent=2)
        
        logger.info(f"✅ 超参数扫描完成，共创建{len(jobs)}个作业")
        
        return jobs

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 10_advanced_finetuning.py single [config_name] [balance_strategy]")
        print("  python 10_advanced_finetuning.py sweep [config1,config2,...]")
        print("  python 10_advanced_finetuning.py analyze")
        print("")
        print("配置选项: conservative, moderate, aggressive, balanced_focus")
        print("平衡策略: oversample, weighted")
        return
    
    action = sys.argv[1]
    finetuner = AdvancedFineTuner()
    
    if action == "single":
        config_name = sys.argv[2] if len(sys.argv) > 2 else 'balanced_focus'
        balance_strategy = sys.argv[3] if len(sys.argv) > 3 else 'weighted'
        
        job_info = finetuner.create_finetuning_job(
            config_name=config_name,
            use_balanced_data=True,
            balance_strategy=balance_strategy
        )
        
        print(f"\\n✅ 单个微调作业创建完成!")
        print(f"作业ID: {job_info['job_id']}")
        print(f"配置: {config_name}")
        
    elif action == "sweep":
        configs = sys.argv[2].split(',') if len(sys.argv) > 2 else None
        jobs = finetuner.run_hyperparameter_sweep(configs)
        
        print(f"\\n✅ 超参数扫描完成!")
        print(f"创建了{len(jobs)}个微调作业")
        for job in jobs:
            print(f"  - {job['config_name']}: {job['job_id']}")
    
    elif action == "analyze":
        # 分析现有训练数据
        distribution_info = finetuner.analyze_data_distribution("finetuning_data/train.jsonl")
        
        print("\\n=== 训练数据分布分析 ===")
        print(f"总样本数: {distribution_info['total_samples']}")
        
        for question, info in distribution_info['question_distributions'].items():
            print(f"\\n{question}:")
            print(f"  样本数: {info['total_samples']}")
            print(f"  分布: {info['percentages']}")
            print(f"  平衡度: {info['balance_score']:.3f}")
            print(f"  是否平衡: {'✅' if info['is_balanced'] else '❌'}")
    
    else:
        print(f"未知操作: {action}")

if __name__ == "__main__":
    main()