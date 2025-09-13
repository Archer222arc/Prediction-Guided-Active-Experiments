#!/usr/bin/env python3
"""
快速检查微调作业状态
"""

import json
from openai import OpenAI

def check_status():
    """检查当前微调作业状态"""
    
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
    
    # 获取当前状态
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    
    print(f"=== 微调作业状态 ===")
    print(f"作业ID: {job_status.id}")
    print(f"状态: {job_status.status}")
    print(f"模型: {job_status.model}")
    print(f"创建时间: {job_status.created_at}")
    
    if job_status.finished_at:
        print(f"完成时间: {job_status.finished_at}")
    
    if job_status.fine_tuned_model:
        print(f"Fine-tuned模型: {job_status.fine_tuned_model}")
    
    if job_status.error:
        print(f"错误: {job_status.error}")
    
    # 显示训练进度（如果有）
    if hasattr(job_status, 'result_files') and job_status.result_files:
        print(f"结果文件: {job_status.result_files}")
    
    return job_status.status

if __name__ == "__main__":
    try:
        status = check_status()
        print(f"\n当前状态: {status}")
        
        if status == "pending":
            print("⏳ 作业正在排队等待处理...")
        elif status == "running":
            print("🔄 微调正在进行中...")
        elif status == "succeeded":
            print("✅ 微调完成!")
        elif status == "failed":
            print("❌ 微调失败")
        elif status == "cancelled":
            print("⚠️ 作业被取消")
            
    except Exception as e:
        print(f"❌ 检查状态失败: {e}")