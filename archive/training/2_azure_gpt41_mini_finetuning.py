#!/usr/bin/env python3
"""
基于Microsoft Learn文档的最小可运行GPT-4.1-mini微调代码
直接使用gpt-4.1-mini-2025-04-14模型
"""

import os
import time
import json
from openai import OpenAI

def test_gpt41_mini_finetuning():
    """测试gpt-4.1-mini-2025-04-14微调"""
    
    # 从配置文件读取Azure信息
    with open("config/azure_models_config.json", 'r') as f:
        config = json.load(f)
    
    azure_config = config["azure_endpoints"]["north_central_us"]
    
    # 1) 连接 Azure OpenAI 数据面（用 v1 路径）
    client = OpenAI(
        api_key=azure_config["api_key"],
        base_url=azure_config["endpoint"].rstrip("/") + "/openai/v1/"
    )
    
    print("=== Azure OpenAI GPT-4.1-mini 微调测试 ===")
    print(f"Endpoint: {azure_config['endpoint']}")
    
    # 2) 上传训练/验证文件（使用之前准备的文件）
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
        """等待文件处理完成"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            file_info = client.files.retrieve(file_id)
            print(f"文件 {file_id} 状态: {file_info.status}")
            if file_info.status == 'processed':
                return True
            elif file_info.status == 'error':
                raise Exception(f"文件处理失败: {file_id}")
            time.sleep(10)
        return False
    
    # 等待两个文件都处理完成
    if not wait_for_file_processing(client, tr.id):
        raise Exception("训练文件处理超时")
    if not wait_for_file_processing(client, vr.id):
        raise Exception("验证文件处理超时")
    
    print("✅ 所有文件处理完成")
    
    # 3) 创建微调任务 —— 关键是这里的 model
    print("\n创建微调作业...")
    job = client.fine_tuning.jobs.create(
        training_file=tr.id,
        validation_file=vr.id,
        model="gpt-4.1-mini-2025-04-14",  # 与 gpt-4.1 用法相同，只是换成 mini 版本
        suffix="npors-survey-predictor",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": -1,  # 自动选择 (~44 for 21,852 samples)
            "learning_rate_multiplier": 0.1  # 更合理的学习率
        }
    )
    print(f"作业ID: {job.id}")
    print(f"模型: {job.model}")
    print(f"状态: {job.status}")
    
    # 保存作业信息
    job_info = {
        "job_id": job.id,
        "model": job.model,
        "training_file": tr.id,
        "validation_file": vr.id,
        "created_at": job.created_at
    }
    
    with open("gpt41_mini_job_info.json", "w") as f:
        json.dump(job_info, f, indent=2)
    
    print(f"\n✅ 微调作业创建成功!")
    print(f"作业信息已保存到: gpt41_mini_job_info.json")
    print("使用 'python 2_azure_gpt41_mini_finetuning.py monitor' 监控进度")
    
    return job.id

def monitor_job():
    """监控微调作业状态"""
    
    # 读取作业信息
    with open("gpt41_mini_job_info.json", "r") as f:
        job_info = json.load(f)
    
    job_id = job_info["job_id"]
    
    # 从配置文件读取Azure信息
    with open("config/azure_models_config.json", 'r') as f:
        config = json.load(f)
    
    azure_config = config["azure_endpoints"]["north_central_us"]
    
    client = OpenAI(
        api_key=azure_config["api_key"],
        base_url=azure_config["endpoint"].rstrip("/") + "/openai/v1/"
    )
    
    print(f"监控微调作业: {job_id}")
    
    # 4) 轮询状态，拿到微调后的模型 ID（用于后续部署）
    while True:
        r = client.fine_tuning.jobs.retrieve(job_id)
        print(f"[{time.strftime('%X')}] 状态: {r.status}")
        
        if r.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(30)  # 每30秒检查一次
    
    if r.status != "succeeded":
        print(f"❌ 微调失败: {r.status}")
        if r.error:
            print(f"错误信息: {r.error}")
        return None
    
    print(f"✅ 微调完成!")
    print(f"Fine-tuned模型: {r.fine_tuned_model}")
    
    # 保存完整结果
    final_result = {
        "job_id": job_id,
        "status": r.status,
        "fine_tuned_model": r.fine_tuned_model,
        "created_at": r.created_at,
        "finished_at": r.finished_at,
        "hyperparameters": r.hyperparameters.__dict__ if r.hyperparameters else None,
        "result_files": r.result_files
    }
    
    with open("gpt41_mini_final_result.json", "w") as f:
        json.dump(final_result, f, indent=2)
    
    print("完整结果已保存到: gpt41_mini_final_result.json")
    
    return r.fine_tuned_model

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 2_azure_gpt41_mini_finetuning.py create    # 创建微调作业")
        print("  python 2_azure_gpt41_mini_finetuning.py monitor   # 监控作业进度")
        return
    
    action = sys.argv[1]
    
    try:
        if action == "create":
            test_gpt41_mini_finetuning()
        elif action == "monitor":
            monitor_job()
        else:
            print(f"未知操作: {action}")
    
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()