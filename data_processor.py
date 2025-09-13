#!/usr/bin/env python3
"""
数字孪生数据集下载与处理脚本
Digital Twin Dataset Download and Processing Script

从Hugging Face下载和处理LLM-Digital-Twin/Twin-2K-500数据集
"""

import os
import json
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_install_datasets():
    """检查并安装datasets库"""
    try:
        from datasets import load_dataset
        logger.info("✅ datasets库已安装")
        return load_dataset
    except ImportError:
        logger.info("正在安装datasets库...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            from datasets import load_dataset
            logger.info("✅ datasets库安装成功")
            return load_dataset
        except Exception as e:
            logger.error(f"❌ datasets库安装失败: {e}")
            raise

def clear_dataset_cache():
    """清理数据集缓存"""
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "LLM-Digital-Twin___parquet"
        if cache_dir.exists():
            logger.info(f"清理缓存目录: {cache_dir}")
            shutil.rmtree(cache_dir)
            return True
    except Exception as e:
        logger.warning(f"清理缓存失败: {e}")
        return False

def load_personas(num_personas: int = 30, force_reload: bool = False) -> Dict[str, str]:
    """
    从Hugging Face加载persona数据
    
    Args:
        num_personas: 加载的persona数量
        force_reload: 是否强制重新下载
    
    Returns:
        Dict[str, str]: persona_id -> persona_summary的映射
    """
    load_dataset = check_and_install_datasets()
    
    logger.info(f"正在加载 {num_personas} 个persona摘要...")
    
    try:
        # 尝试加载数据集
        dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", 'full_persona', split='data')
        logger.info("✅ 数据集加载成功")
        
    except Exception as e:
        logger.warning(f"⚠️ 加载数据集失败: {type(e).__name__}: {str(e)}")
        
        if force_reload or input("是否清理缓存并重新下载？(y/n): ").lower() == 'y':
            clear_dataset_cache()
            
            try:
                dataset = load_dataset(
                    "LLM-Digital-Twin/Twin-2K-500", 
                    'full_persona', 
                    split='data', 
                    download_mode='force_redownload'
                )
                logger.info("✅ 重新下载成功")
            except Exception as e2:
                logger.error(f"❌ 重新下载仍然失败: {type(e2).__name__}: {str(e2)}")
                logger.error("请手动下载数据集: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500")
                raise e2
        else:
            raise e
    
    # 提取personas
    personas = {}
    pids = dataset["pid"]
    persona_summaries = dataset["persona_summary"]
    
    # 确保不超过可用数据量
    max_available = len(pids)
    actual_num = min(num_personas, max_available)
    
    logger.info(f"数据集包含 {max_available} 个persona，将加载 {actual_num} 个")
    
    # 加载指定数量的personas
    for i in range(actual_num):
        pid = pids[i]
        summary = persona_summaries[i]
        
        if summary is not None:
            personas[f"pid_{pid}"] = summary
    
    logger.info(f"✅ 成功加载 {len(personas)} 个personas")
    return personas

def load_ground_truth_data(force_reload: bool = False) -> List:
    """
    加载ground truth数据
    
    Args:
        force_reload: 是否强制重新下载
    
    Returns:
        List: ground truth响应数据
    """
    load_dataset = check_and_install_datasets()
    
    logger.info("正在加载ground truth数据...")
    
    try:
        wave_split = load_dataset("LLM-Digital-Twin/Twin-2K-500", "wave_split")
        ground_truth = wave_split["data"]["wave4_Q_wave4_A"]
        
        logger.info(f"✅ 成功加载 {len(ground_truth)} 条ground truth记录")
        return ground_truth
        
    except Exception as e:
        logger.error(f"❌ 加载ground truth数据失败: {e}")
        if force_reload:
            clear_dataset_cache()
            try:
                wave_split = load_dataset(
                    "LLM-Digital-Twin/Twin-2K-500", 
                    "wave_split",
                    download_mode='force_redownload'
                )
                ground_truth = wave_split["data"]["wave4_Q_wave4_A"]
                logger.info(f"✅ 重新下载成功，加载 {len(ground_truth)} 条记录")
                return ground_truth
            except Exception as e2:
                logger.error(f"❌ 重新下载失败: {e2}")
                raise e2
        else:
            raise e

def save_ground_truth_to_json(ground_truth: List, filename: str = "ground_truth_output.json") -> None:
    """
    保存ground truth数据到JSON文件
    
    Args:
        ground_truth: ground truth数据
        filename: 输出文件名
    """
    logger.info(f"正在保存ground truth数据到 {filename}...")
    
    parsed = []
    for i, entry in enumerate(ground_truth):
        try:
            if isinstance(entry, str):
                parsed.append(json.loads(entry))
            else:
                parsed.append(entry)
        except Exception as e:
            logger.warning(f"解析第 {i} 条记录失败: {e}")
            parsed.append({"error": str(e), "raw": entry})

    # 确保输出目录存在
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 已保存到 {filename}")

def save_personas_to_json(personas: Dict[str, str], filename: str = "personas_output.json") -> None:
    """
    保存personas数据到JSON文件
    
    Args:
        personas: personas字典
        filename: 输出文件名
    """
    logger.info(f"正在保存 {len(personas)} 个personas到 {filename}...")
    
    # 确保输出目录存在
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 已保存到 {filename}")

def analyze_persona_sample(personas: Dict[str, str], show_first_n: int = 3) -> None:
    """
    分析并显示persona样本
    
    Args:
        personas: personas字典
        show_first_n: 显示前N个persona的数量
    """
    if not personas:
        logger.warning("没有可分析的persona数据")
        return
    
    logger.info("=" * 50)
    logger.info("PERSONA 样本分析")
    logger.info("=" * 50)
    
    # 显示总体统计
    logger.info(f"总persona数量: {len(personas)}")
    
    # 显示前几个persona的摘要
    for i, (pid, summary) in enumerate(list(personas.items())[:show_first_n]):
        logger.info(f"\n【{pid}】")
        logger.info(f"摘要长度: {len(summary)} 字符")
        logger.info(f"前500字符: {summary[:500]}...")
        
        # 简单分析关键信息
        demographics = extract_demographics_info(summary)
        if demographics:
            logger.info(f"关键信息: {demographics}")

def extract_demographics_info(persona_text: str) -> Dict[str, str]:
    """
    从persona文本中提取关键人口统计学信息
    
    Args:
        persona_text: persona描述文本
        
    Returns:
        Dict[str, str]: 提取的关键信息
    """
    info = {}
    
    # 简单的关键词匹配提取
    lines = persona_text.split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # 提取关键字段
            if any(keyword in key.lower() for keyword in ['gender', 'age', 'education', 'income', 'political']):
                info[key] = value
    
    return info

def validate_data_integrity(personas: Dict[str, str], ground_truth: List) -> bool:
    """
    验证数据完整性
    
    Args:
        personas: personas数据
        ground_truth: ground truth数据
        
    Returns:
        bool: 验证是否通过
    """
    logger.info("正在验证数据完整性...")
    
    issues = []
    
    # 检查personas
    if not personas:
        issues.append("Personas数据为空")
    else:
        empty_personas = [pid for pid, summary in personas.items() if not summary or len(summary.strip()) < 50]
        if empty_personas:
            issues.append(f"发现 {len(empty_personas)} 个空或过短的persona: {empty_personas[:5]}")
    
    # 检查ground truth
    if not ground_truth:
        issues.append("Ground truth数据为空")
    else:
        try:
            # 尝试解析第一条记录
            first_record = ground_truth[0]
            if isinstance(first_record, str):
                json.loads(first_record)
            logger.info(f"Ground truth包含 {len(ground_truth)} 条记录")
        except Exception as e:
            issues.append(f"Ground truth数据格式异常: {e}")
    
    if issues:
        logger.warning("数据验证发现问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("✅ 数据验证通过")
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数字孪生数据集处理工具')
    parser.add_argument('--num-personas', type=int, default=30, help='加载的persona数量')
    parser.add_argument('--output-dir', default='./data', help='输出目录')
    parser.add_argument('--force-reload', action='store_true', help='强制重新下载')
    parser.add_argument('--skip-ground-truth', action='store_true', help='跳过ground truth下载')
    parser.add_argument('--analyze-only', action='store_true', help='仅分析现有数据，不下载')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if not args.analyze_only:
            # 1. 加载personas
            logger.info("开始处理数字孪生数据集...")
            personas = load_personas(args.num_personas, args.force_reload)
            
            # 保存personas
            personas_file = output_dir / "personas_output.json"
            save_personas_to_json(personas, str(personas_file))
            
            # 2. 加载ground truth (可选)
            if not args.skip_ground_truth:
                ground_truth = load_ground_truth_data(args.force_reload)
                
                # 保存ground truth
                gt_file = output_dir / "ground_truth_output.json"
                save_ground_truth_to_json(ground_truth, str(gt_file))
            else:
                logger.info("跳过ground truth数据下载")
                ground_truth = []
            
            # 3. 验证数据
            validate_data_integrity(personas, ground_truth)
            
            # 4. 分析样本
            analyze_persona_sample(personas)
        
        else:
            # 仅分析模式
            logger.info("分析模式：检查现有数据文件...")
            
            personas_file = output_dir / "personas_output.json"
            if personas_file.exists():
                with open(personas_file, 'r', encoding='utf-8') as f:
                    personas = json.load(f)
                analyze_persona_sample(personas)
            else:
                logger.warning(f"未找到personas文件: {personas_file}")
        
        logger.info("🎉 数据处理完成!")
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()