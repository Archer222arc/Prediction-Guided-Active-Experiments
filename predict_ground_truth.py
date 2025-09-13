#!/usr/bin/env python3
"""
针对ground_truth_output.json生成数字孪生预测
Generate digital twin predictions for ground truth personas
"""

import json
import pandas as pd
import time
import logging
from typing import Dict, List, Optional
from digital_twin_prediction import DigitalTwinPredictor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroundTruthPredictor:
    """基于Ground Truth数据的预测器"""
    
    def __init__(self):
        self.predictor = DigitalTwinPredictor()
        self.ground_truth_file = "./data/ground_truth_output.json"
        
    def load_ground_truth_personas(self) -> Dict[str, str]:
        """从ground truth文件中加载persona信息"""
        
        logger.info(f"加载ground truth数据: {self.ground_truth_file}")
        
        try:
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            logger.info(f"成功加载 {len(ground_truth)} 条ground truth记录")
            
            # 提取personas（每条记录对应一个persona的完整调查响应）
            personas = {}
            
            for i, entry in enumerate(ground_truth):
                # 检查数据结构
                if isinstance(entry, list) and len(entry) > 0:
                    # 取第一个元素（通常包含persona信息）
                    persona_data = entry[0]
                    
                    if isinstance(persona_data, dict) and 'Questions' in persona_data:
                        # 这里需要根据实际数据结构提取persona描述
                        # 由于ground truth主要包含响应而非persona描述，
                        # 我们需要从原始Twin-2K-500数据集匹配
                        persona_id = f"gt_persona_{i+1}"
                        
                        # 暂时使用占位符，实际使用中需要匹配原始persona数据
                        persona_text = f"Ground truth persona {i+1} (需要匹配原始数据集获取完整描述)"
                        personas[persona_id] = persona_text
                
                # 限制处理数量以避免过大的计算量
                if len(personas) >= 100:  # 先处理前100个进行测试
                    logger.info(f"限制处理前 {len(personas)} 个personas进行测试")
                    break
            
            logger.info(f"提取了 {len(personas)} 个personas")
            return personas
            
        except Exception as e:
            logger.error(f"加载ground truth数据失败: {e}")
            return {}
    
    def load_original_personas_for_ground_truth(self, num_personas: int = 100) -> Dict[str, str]:
        """
        加载原始personas数据作为ground truth的代理
        由于ground truth文件主要包含响应数据，我们使用原始personas
        """
        from data_processor import load_personas
        
        if num_personas == -1:
            logger.info("加载全部2058个原始personas数据")
            # 加载全部personas
            try:
                # 直接调用datasets加载全部数据
                from datasets import load_dataset
                dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", 'full_persona', split='data')
                
                personas = {}
                pids = dataset["pid"]
                persona_summaries = dataset["persona_summary"]
                
                # 加载所有personas
                for i in range(len(pids)):
                    pid = pids[i]
                    summary = persona_summaries[i]
                    
                    if summary is not None:
                        gt_id = f"gt_pid_{pid}"
                        personas[gt_id] = summary
                
                logger.info(f"成功加载全部 {len(personas)} 个personas")
                return personas
                
            except Exception as e:
                logger.error(f"加载全部personas失败: {e}")
                return {}
        else:
            logger.info(f"加载原始personas数据作为ground truth代理 (数量: {num_personas})")
            
            try:
                personas = load_personas(num_personas)
                
                # 重新命名为ground truth格式
                gt_personas = {}
                for i, (original_id, persona_text) in enumerate(personas.items()):
                    gt_id = f"gt_{original_id}"
                    gt_personas[gt_id] = persona_text
                
                logger.info(f"准备了 {len(gt_personas)} 个ground truth personas")
                return gt_personas
                
            except Exception as e:
                logger.error(f"加载原始personas失败: {e}")
                return {}
    
    def run_predictions(self, method: str, num_personas: int = 30, max_workers: int = 8, max_retries: int = 5):
        """运行预测并保存结果"""
        
        if num_personas == -1:
            logger.info(f"开始全量ground truth预测 (方法: {method}, 全部2058个personas, 线程: {max_workers}, 重试: {max_retries})")
            logger.info("⚠️ 全量处理将需要2-4小时，请确保网络稳定")
        else:
            logger.info(f"开始ground truth预测 (方法: {method}, personas: {num_personas}, 线程: {max_workers}, 重试: {max_retries})")
        
        # 加载personas
        personas = self.load_original_personas_for_ground_truth(num_personas)
        
        if not personas:
            logger.error("未能加载personas数据")
            return None
        
        actual_count = len(personas)
        total_api_calls = actual_count * 10  # 每个persona 10个问题
        
        logger.info(f"实际加载 {actual_count} 个personas，预计 {total_api_calls} 次API调用")
        
        # 全量处理的特殊提示
        if actual_count > 500:
            logger.info("🚀 大规模处理开始，建议监控进度...")
            estimated_time = (total_api_calls / max_workers) / 60  # 粗略估计（分钟）
            logger.info(f"预估完成时间: {estimated_time:.0f} 分钟")
        
        # 执行预测
        start_time = time.time()
        df_results = self.predictor.process_personas_dataset(personas, method, max_workers, max_retries)
        end_time = time.time()
        
        # 保存结果（使用特定的命名格式）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if actual_count >= 2000:
            filename = f'ground_truth_predictions_FULL_{method}_{timestamp}.csv'
        else:
            filename = f'ground_truth_predictions_{method}_{timestamp}.csv'
        
        output_file = self.predictor.save_results(df_results, personas, method, filename)
        
        elapsed_minutes = (end_time - start_time) / 60
        logger.info(f"Ground truth预测完成，耗时: {elapsed_minutes:.1f} 分钟")
        
        # 全量处理的统计报告
        if actual_count > 100:
            success_count = sum(1 for _, row in df_results.iterrows() 
                              if any(pd.notna(row[col]) for col in row.index if col.endswith('_LLM')))
            success_rate = success_count / actual_count if actual_count > 0 else 0
            avg_time_per_persona = elapsed_minutes / actual_count if actual_count > 0 else 0
            
            logger.info(f"📊 处理统计:")
            logger.info(f"  - 成功处理: {success_count}/{actual_count} ({success_rate:.1%})")
            logger.info(f"  - 平均每个persona: {avg_time_per_persona:.2f} 分钟")
            logger.info(f"  - API调用效率: {total_api_calls/elapsed_minutes:.1f} 次/分钟")
        
        return output_file

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python predict_ground_truth.py base [num_personas] [max_workers] [max_retries]")
        print("  python predict_ground_truth.py cot [num_personas] [max_workers] [max_retries]")
        print("  python predict_ground_truth.py both [num_personas] [max_workers] [max_retries]")
        print("")
        print("参数:")
        print("  method:")
        print("    base: 基础方法预测")
        print("    cot: Chain-of-Thought方法预测") 
        print("    both: 依次运行base和cot两种方法")
        print("  num_personas: 处理的persona数量")
        print("    数字: 指定数量 (如30, 100)")
        print("    -1: 处理全部2058个personas")
        print("    默认: 30")
        print("  max_workers: 线程数 (默认: 8, 已优化)")
        print("  max_retries: 重试次数 (默认: 5)")
        print("")
        print("性能优化:")
        print("  - 移除了固定sleep延迟")
        print("  - 增加了指数退避重试机制")
        print("  - 提高了默认并发数")
        print("")
        print("输出文件:")
        print("  ground_truth_predictions_base_YYYYMMDD_HHMMSS.csv")
        print("  ground_truth_predictions_cot_YYYYMMDD_HHMMSS.csv")
        print("")
        print("全量处理说明 (num_personas = -1):")
        print("  - 总计: 2058个personas × 10个问题 = 20,580次API调用")
        print("  - 预计时间: 2-4小时 (取决于并发数和网络)")
        print("  - 建议配置: python predict_ground_truth.py both -1 15 3")
        print("  - 文件大小: ~1-2MB CSV文件")
        return
    
    method = sys.argv[1]
    if method not in ['base', 'cot', 'both']:
        print("错误: method 必须是 'base', 'cot' 或 'both'")
        return
    
    num_personas = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_retries = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    gt_predictor = GroundTruthPredictor()
    
    if method == "both":
        # 依次运行两种方法
        logger.info("开始运行base和cot两种方法的ground truth预测")
        
        # 运行base方法
        logger.info("\n" + "="*50)
        logger.info("第1步: 运行BASE方法")
        logger.info("="*50)
        base_output = gt_predictor.run_predictions("base", num_personas, max_workers, max_retries)
        
        # 根据处理规模调整等待时间
        if num_personas == -1 or num_personas > 500:
            logger.info("全量处理完成，等待5分钟后开始CoT预测...")
            time.sleep(300)  # 全量处理后等待5分钟
        else:
            logger.info("等待30秒后开始CoT预测...")
            time.sleep(30)
        
        # 运行cot方法
        logger.info("\n" + "="*50)
        logger.info("第2步: 运行COT方法")
        logger.info("="*50)
        cot_output = gt_predictor.run_predictions("cot", num_personas, max_workers, max_retries)
        
        print(f"\n✅ 两种方法的ground truth预测全部完成!")
        print(f"Base方法结果: {base_output}")
        print(f"CoT方法结果: {cot_output}")
        
    else:
        # 运行单个方法
        output_file = gt_predictor.run_predictions(method, num_personas, max_workers, max_retries)
        
        print(f"\n✅ Ground truth {method.upper()}方法预测完成!")
        print(f"处理的personas: {num_personas}")
        print(f"并发线程: {max_workers}")
        print(f"重试次数: {max_retries}")
        print(f"结果文件: {output_file}")

if __name__ == "__main__":
    main()