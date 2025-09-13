#!/usr/bin/env python3
"""
优化版 GPT-4.1-mini 微调脚本
基于专家建议优化超参数和训练监控
"""

import os
import time
import json
from openai import OpenAI

def create_optimized_finetuning_job():
    """创建优化的微调作业"""
    
    # 从配置文件读取Azure信息
    with open("config/azure_models_config.json", 'r') as f:
        config = json.load(f)
    
    azure_config = config["azure_endpoints"]["north_central_us"]
    
    client = OpenAI(
        api_key=azure_config["api_key"],
        base_url=azure_config["endpoint"].rstrip("/") + "/openai/v1/"
    )
    
    print("=== 优化版 Azure OpenAI GPT-4.1-mini 微调 ===")
    print(f"数据集大小: 21,852 训练样本 + 5,463 验证样本")
    
    # 上传文件
    print("\n上传训练文件...")
    tr = client.files.create(
        file=open("finetuning_data/train.jsonl", "rb"), 
        purpose="fine-tune"
    )
    print(f"训练文件ID: {tr.id}")
    
    print("上传验证文件...")
    vr = client.files.create(
        file=open("finetuning_data/validation.jsonl", "rb"), 
        purpose="fine-tune"
    )
    print(f"验证文件ID: {vr.id}")
    
    # 等待文件处理完成
    print("\n等待文件处理完成...")
    def wait_for_file_processing(client, file_id, max_wait=300):
        start_time = time.time()
        while time.time() - start_time < max_wait:
            file_info = client.files.retrieve(file_id)
            if file_info.status == 'processed':
                return True
            elif file_info.status == 'error':
                raise Exception(f"文件处理失败: {file_id}")
            time.sleep(5)
        return False
    
    if not wait_for_file_processing(client, tr.id):
        raise Exception("训练文件处理超时")
    if not wait_for_file_processing(client, vr.id):
        raise Exception("验证文件处理超时")
    
    print("✅ 文件处理完成")
    
    # 优化的超参数设置
    optimized_hyperparams = {
        "n_epochs": 3,
        "batch_size": -1,  # 自动选择，预计 ~44 (21,852 * 0.2% = 44)
        "learning_rate_multiplier": 0.1  # 适中的学习率
    }
    
    print(f"\n=== 优化的超参数设置 ===")
    print(f"Epochs: {optimized_hyperparams['n_epochs']}")
    print(f"Batch Size: 自动 (预计 ~44)")
    print(f"Learning Rate Multiplier: {optimized_hyperparams['learning_rate_multiplier']}")
    print(f"预计总步数: ~1,491 (vs 之前的 65,556)")
    
    # 创建微调作业
    print("\n创建优化微调作业...")
    job = client.fine_tuning.jobs.create(
        training_file=tr.id,
        validation_file=vr.id,
        model="gpt-4.1-mini-2025-04-14",
        suffix="npors-predictor-optimized",
        hyperparameters=optimized_hyperparams
    )
    
    print(f"✅ 优化微调作业创建成功!")
    print(f"作业ID: {job.id}")
    print(f"模型: {job.model}")
    print(f"状态: {job.status}")
    
    # 保存作业信息
    job_info = {
        "job_id": job.id,
        "model": job.model,
        "training_file": tr.id,
        "validation_file": vr.id,
        "hyperparameters": optimized_hyperparams,
        "created_at": job.created_at,
        "expected_steps": "~1,491",
        "optimization_notes": "使用自动batch_size和优化学习率"
    }
    
    with open("optimized_finetuning_job.json", "w") as f:
        json.dump(job_info, f, indent=2)
    
    print(f"作业信息保存到: optimized_finetuning_job.json")
    
    return job.id

def monitor_optimized_job():
    """监控优化版微调作业，重点关注训练指标"""
    
    with open("optimized_finetuning_job.json", "r") as f:
        job_info = json.load(f)
    
    job_id = job_info["job_id"]
    
    with open("config/azure_models_config.json", 'r') as f:
        config = json.load(f)
    
    azure_config = config["azure_endpoints"]["north_central_us"]
    
    client = OpenAI(
        api_key=azure_config["api_key"],
        base_url=azure_config["endpoint"].rstrip("/") + "/openai/v1/"
    )
    
    print(f"=== 监控优化微调作业 ===")
    print(f"作业ID: {job_id}")
    print(f"预计步数: ~1,491 (vs 之前65,556)")
    
    while True:
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        current_time = time.strftime('%X')
        
        print(f"[{current_time}] 状态: {job_status.status}")
        
        # 显示训练进度（如果有）
        if hasattr(job_status, 'trained_tokens') and job_status.trained_tokens:
            print(f"  已训练tokens: {job_status.trained_tokens:,}")
        
        if job_status.status in ("succeeded", "failed", "cancelled"):
            break
            
        if job_status.status == "running":
            print("  🔄 训练中... (使用优化参数，预计更快完成)")
            
        time.sleep(60)  # 每分钟检查一次
    
    if job_status.status == "succeeded":
        print(f"✅ 优化微调完成!")
        print(f"Fine-tuned模型: {job_status.fine_tuned_model}")
        
        # 保存最终结果
        final_result = {
            "job_id": job_id,
            "status": job_status.status,
            "fine_tuned_model": job_status.fine_tuned_model,
            "created_at": job_status.created_at,
            "finished_at": job_status.finished_at,
            "hyperparameters": job_status.hyperparameters.__dict__ if job_status.hyperparameters else None,
            "result_files": job_status.result_files,
            "optimization_used": True
        }
        
        with open("optimized_finetuning_result.json", "w") as f:
            json.dump(final_result, f, indent=2)
        
        print("结果保存到: optimized_finetuning_result.json")
        
        # 下载训练结果文件
        if job_status.result_files:
            print("\n下载训练结果...")
            for file_id in job_status.result_files:
                try:
                    content = client.files.content(file_id)
                    filename = f"training_results_optimized_{file_id}.csv"
                    with open(filename, "wb") as f:
                        f.write(content.read())
                    print(f"训练结果已下载: {filename}")
                    print("⚠️ 检查 train_loss/valid_loss 曲线以评估训练质量")
                except Exception as e:
                    print(f"下载结果文件失败: {e}")
        
        return job_status.fine_tuned_model
        
    else:
        print(f"❌ 训练失败: {job_status.status}")
        if job_status.error:
            print(f"错误: {job_status.error}")
        return None

def compare_approaches():
    """比较原始方法vs优化方法"""
    print("=== 训练方法对比 ===")
    print("原始设置:")
    print("  batch_size: 1")
    print("  learning_rate_multiplier: 1.0") 
    print("  预计步数: 65,556 (21,852 × 3)")
    print("  训练时间: 长")
    print()
    print("优化设置:")
    print("  batch_size: -1 (自动, ~44)")
    print("  learning_rate_multiplier: 0.1")
    print("  预计步数: ~1,491 (⌈21,852/44⌉ × 3)")
    print("  训练时间: 大幅缩短")
    print("  收敛性: 更好")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 2b_optimized_gpt41_mini_finetuning.py create     # 创建优化微调作业")
        print("  python 2b_optimized_gpt41_mini_finetuning.py monitor    # 监控优化作业")
        print("  python 2b_optimized_gpt41_mini_finetuning.py compare    # 对比训练方法")
        return
    
    action = sys.argv[1]
    
    try:
        if action == "create":
            create_optimized_finetuning_job()
        elif action == "monitor":
            monitor_optimized_job()
        elif action == "compare":
            compare_approaches()
        else:
            print(f"未知操作: {action}")
    
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()