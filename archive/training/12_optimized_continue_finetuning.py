#!/usr/bin/env python3
"""
进一步优化的继续微调脚本 - 基于训练分析结果的最优参数配置
解决系统性偏差，平衡稳定性与性能
"""

import os
import time
import json
from openai import OpenAI
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedContinuedFineTuner:
    """优化版继续微调器 - 基于训练曲线分析的最佳实践"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        azure_config = config["azure_endpoints"]["north_central_us"]
        
        self.client = OpenAI(
            api_key=azure_config["api_key"],
            base_url=azure_config["endpoint"].rstrip("/") + "/openai/v1/"
        )
        
        # 优化的超参数配置组合
        self.hyperparameter_configs = {
            'final_optimal': {
                'n_epochs': 1,
                'batch_size': 64,
                'learning_rate_multiplier': 0.6,  # 基于分析的最优平衡点
                'description': '最终优化配置 - 基于4轮训练分析的最佳参数'
            },
            'balanced_optimal': {
                'n_epochs': 2,
                'batch_size': 64,
                'learning_rate_multiplier': 0.75,  # 在0.5和1.5之间的平衡点
                'description': '平衡最优配置 - 稳定性与性能并重'
            },
            'conservative_stable': {
                'n_epochs': 3,
                'batch_size': 64,
                'learning_rate_multiplier': 0.5,
                'description': '保守稳定配置 - 确保训练稳定'
            },
            'performance_focused': {
                'n_epochs': 2,
                'batch_size': 48,  # 稍小批量大小以提高学习能力
                'learning_rate_multiplier': 1.0,
                'description': '性能优先配置 - 追求更高准确率'
            },
            'extended_training': {
                'n_epochs': 4,
                'batch_size': 64,
                'learning_rate_multiplier': 0.6,
                'description': '延长训练配置 - 更多轮次深度学习'
            }
        }
        
        logger.info("优化版继续微调器初始化完成")
    
    def find_best_checkpoint(self, prev_job_id: str):
        """找到最佳检查点 - 优化版"""
        
        logger.info(f"分析作业 {prev_job_id} 的检查点...")
        
        try:
            # 列出检查点
            ckpt_list = self.client.fine_tuning.jobs.checkpoints.list(prev_job_id)
            
            if not ckpt_list.data:
                raise RuntimeError("没有找到检查点")
            
            # 定义可能的指标键名
            valid_loss_keys = ["full_valid_loss", "valid_loss", "eval_loss"]
            valid_acc_keys = ["full_valid_mean_token_accuracy", "valid_mean_token_accuracy", 
                             "valid_accuracy", "eval_accuracy"]
            train_loss_keys = ["full_train_loss", "train_loss", "training_loss"]
            
            best = None
            
            for item in ckpt_list.data:
                # 获取模型名
                model_name = self._get_checkpoint_model_name(item)
                step = getattr(item, "step_number", 0)
                
                # 转换metrics为字典
                metrics_dict = {}
                if hasattr(item, 'metrics'):
                    if hasattr(item.metrics, '__dict__'):
                        metrics_dict = item.metrics.__dict__
                    elif hasattr(item.metrics, 'model_dump'):
                        metrics_dict = item.metrics.model_dump()
                    else:
                        try:
                            metrics_dict = dict(item.metrics)
                        except:
                            continue
                
                # 优先选择验证损失最小的
                score = None
                metric_name = None
                for key in valid_loss_keys:
                    if key in metrics_dict and metrics_dict[key] is not None:
                        score = metrics_dict[key]
                        metric_name = key
                        break
                
                # 如果没有验证损失，尝试验证准确率最大的
                if score is None:
                    for key in valid_acc_keys:
                        if key in metrics_dict and metrics_dict[key] is not None:
                            score = -metrics_dict[key]  # 负号使得最大变最小
                            metric_name = key + " (最大)"
                            break
                
                # 最后尝试训练损失
                if score is None:
                    for key in train_loss_keys:
                        if key in metrics_dict and metrics_dict[key] is not None:
                            score = metrics_dict[key]
                            metric_name = key
                            break
                
                if score is not None:
                    if best is None or score < best[0] or (score == best[0] and step > best[2]):
                        best = (score, model_name, step, metric_name)
            
            if best is None:
                raise RuntimeError("没有找到有效的指标")
            
            best_score, best_checkpoint, best_step, best_metric = best
            
            logger.info(f"✅ 找到最佳检查点:")
            logger.info(f"  步数: {best_step}")
            logger.info(f"  指标: {best_metric} = {abs(best_score):.6f}")
            logger.info(f"  模型名: {best_checkpoint}")
            
            return best_checkpoint, abs(best_score), best_step
            
        except Exception as e:
            logger.error(f"获取检查点失败: {e}")
            # 回退到最终模型
            logger.info("回退到使用最终微调模型...")
            job = self.client.fine_tuning.jobs.retrieve(prev_job_id)
            return job.fine_tuned_model, None, None
    
    def _get_checkpoint_model_name(self, item):
        """从检查点项中获取模型名"""
        for key in ("fine_tuned_model_checkpoint", "checkpoint", "fine_tuned_model", "model"):
            val = getattr(item, key, None)
            if isinstance(val, str) and len(val) > 0:
                return val
            # 尝试字典化后取
            if hasattr(item, "model_dump"):
                d = item.model_dump()
                if key in d and isinstance(d[key], str) and d[key]:
                    return d[key]
        raise RuntimeError("无法从checkpoint项提取模型名")
    
    def create_optimized_training_data(self, base_file: str, output_file: str, strategy: str = 'moderate'):
        """创建优化的训练数据 - 基于分析结果的精准平衡"""
        
        logger.info(f"创建优化训练数据，策略: {strategy}...")
        
        with open(base_file, 'r') as f:
            lines = f.readlines()
        
        balanced_lines = []
        
        # 不同策略的权重配置
        weight_strategies = {
            'gentle': {  # 温和平衡
                'UNITY: 1': 1.5,
                'GAMBLERESTR: 1': 1.5,
                'GAMBLERESTR: 3': 1.5,
                'description': '温和平衡 - 轻微增强少数类'
            },
            'moderate': {  # 适中平衡（基于成功的测试）
                'UNITY: 1': 2.0,
                'GAMBLERESTR: 1': 2.0,
                'GAMBLERESTR: 3': 2.0,
                'description': '适中平衡 - 已验证有效的策略'
            },
            'focused': {  # 聚焦平衡
                'UNITY: 1': 2.5,
                'GAMBLERESTR: 1': 2.5,
                'GAMBLERESTR: 3': 2.5,
                'description': '聚焦平衡 - 针对性增强'
            }
        }
        
        weights = weight_strategies.get(strategy, weight_strategies['moderate'])
        logger.info(f"使用权重策略: {weights['description']}")
        
        for line in lines:
            try:
                data = json.loads(line)
                repeat_count = 1
                
                if 'messages' in data:
                    assistant_msg = None
                    for msg in data['messages']:
                        if msg['role'] == 'assistant':
                            assistant_msg = msg['content']
                            break
                    
                    if assistant_msg:
                        for pattern, weight in weights.items():
                            if isinstance(weight, float) and pattern in assistant_msg:
                                repeat_count = int(weight)
                                break
                
                for _ in range(repeat_count):
                    balanced_lines.append(line)
                    
            except:
                balanced_lines.append(line)
        
        # 随机打乱
        random.shuffle(balanced_lines)
        
        with open(output_file, 'w') as f:
            f.writelines(balanced_lines)
        
        logger.info(f"优化数据创建完成:")
        logger.info(f"  原始: {len(lines)} 样本")
        logger.info(f"  平衡后: {len(balanced_lines)} 样本")
        logger.info(f"  增加倍率: {len(balanced_lines)/len(lines):.2f}x")
        
        return output_file
    
    def create_enhanced_validation_set(self, base_train_file: str, output_validation_file: str, 
                                     validation_ratio: float = 0.18):
        """创建增强的验证集 - 提高验证数据质量"""
        
        logger.info(f"创建增强验证集，比例: {validation_ratio:.1%}...")
        
        with open(base_train_file, 'r') as f:
            all_lines = f.readlines()
        
        # 分层采样确保验证集代表性
        question_lines = {'ECON1MOD': [], 'UNITY': [], 'GPT1': [], 'MOREGUNIMPACT': [], 'GAMBLERESTR': []}
        other_lines = []
        
        for line in all_lines:
            try:
                data = json.loads(line)
                if 'messages' in data:
                    found = False
                    for msg in data['messages']:
                        if msg['role'] == 'assistant':
                            for question in question_lines.keys():
                                if question in msg['content']:
                                    question_lines[question].append(line)
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        other_lines.append(line)
                else:
                    other_lines.append(line)
            except:
                other_lines.append(line)
        
        # 从每个问题类型中分层抽样
        val_lines = []
        remaining_lines = []
        
        for question, lines in question_lines.items():
            if lines:
                random.shuffle(lines)
                val_size = max(1, int(len(lines) * validation_ratio))
                val_lines.extend(lines[:val_size])
                remaining_lines.extend(lines[val_size:])
        
        # 处理其他样本
        if other_lines:
            random.shuffle(other_lines)
            val_size = int(len(other_lines) * validation_ratio)
            val_lines.extend(other_lines[:val_size])
            remaining_lines.extend(other_lines[val_size:])
        
        # 保存文件
        with open(output_validation_file, 'w') as f:
            f.writelines(val_lines)
        
        with open(base_train_file, 'w') as f:
            f.writelines(remaining_lines)
        
        logger.info(f"增强验证集创建完成:")
        logger.info(f"  训练样本: {len(remaining_lines)}")
        logger.info(f"  验证样本: {len(val_lines)}")
        logger.info(f"  验证比例: {len(val_lines)/(len(remaining_lines)+len(val_lines)):.1%}")
        
        return output_validation_file
    
    def continue_training_optimized(self, prev_job_id: str, config_name: str = 'balanced_optimal',
                                  balance_strategy: str = 'moderate', use_best_checkpoint: bool = True):
        """优化版继续训练"""
        
        logger.info(f"开始优化版继续训练...")
        logger.info(f"配置: {config_name}, 平衡策略: {balance_strategy}")
        
        # 获取超参数配置
        if config_name not in self.hyperparameter_configs:
            logger.warning(f"未知配置 {config_name}，使用默认 balanced_optimal")
            config_name = 'balanced_optimal'
        
        hyperparams = self.hyperparameter_configs[config_name]
        logger.info(f"配置描述: {hyperparams['description']}")
        
        # 确定基础模型
        if use_best_checkpoint:
            base_model, best_score, best_step = self.find_best_checkpoint(prev_job_id)
            suffix = f"opt-{config_name}-step{best_step}"
        else:
            job = self.client.fine_tuning.jobs.retrieve(prev_job_id)
            base_model = job.fine_tuned_model
            suffix = f"opt-{config_name}-final"
        
        logger.info(f"基础模型: {base_model}")
        
        # 创建优化的训练数据
        balanced_train_file = self.create_optimized_training_data(
            "finetuning_data/train.jsonl",
            f"finetuning_data/train_optimized_{balance_strategy}.jsonl",
            strategy=balance_strategy
        )
        
        # 创建增强的验证集
        validation_file = self.create_enhanced_validation_set(
            balanced_train_file,
            f"finetuning_data/validation_enhanced_{config_name}.jsonl",
            validation_ratio=0.18  # 稍微增加验证集比例
        )
        
        # 上传文件
        logger.info("上传优化训练文件...")
        tr = self.client.files.create(
            file=open(balanced_train_file, "rb"), 
            purpose="fine-tune"
        )
        
        logger.info("上传增强验证文件...")
        vr = self.client.files.create(
            file=open(validation_file, "rb"), 
            purpose="fine-tune"
        )
        
        # 等待文件处理
        self._wait_for_file_processing(tr.id)
        self._wait_for_file_processing(vr.id)
        
        # 优化的超参数配置
        final_hyperparams = {
            "batch_size": hyperparams['batch_size'],
            "learning_rate_multiplier": hyperparams['learning_rate_multiplier'],
            "n_epochs": hyperparams['n_epochs'],
            "seed": 42
        }
        
        logger.info("创建优化继续训练作业...")
        logger.info(f"超参数: {final_hyperparams}")
        
        job = self.client.fine_tuning.jobs.create(
            model=base_model,
            training_file=tr.id,
            validation_file=vr.id,
            hyperparameters=final_hyperparams,
            suffix=suffix
        )
        
        # 保存作业信息
        job_info = {
            "continue_job_id": job.id,
            "base_job_id": prev_job_id,
            "base_model": base_model,
            "config_name": config_name,
            "balance_strategy": balance_strategy,
            "use_best_checkpoint": use_best_checkpoint,
            "hyperparameters": final_hyperparams,
            "training_file": tr.id,
            "validation_file": vr.id,
            "created_at": job.created_at,
            "status": job.status,
            "training_samples": len(open(balanced_train_file).readlines()),
            "validation_samples": len(open(validation_file).readlines()),
            "optimization_notes": "基于训练曲线分析的优化版本"
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_file = f"optimized_continue_finetuning_{config_name}_{timestamp}.json"
        
        with open(job_file, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        logger.info(f"✅ 优化继续训练作业创建成功!")
        logger.info(f"新作业ID: {job.id}")
        logger.info(f"基于: {base_model}")
        logger.info(f"配置: {config_name} ({hyperparams['description']})")
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
    
    def run_optimized_sweep(self, prev_job_id: str, configs: list = None):
        """运行优化的超参数扫描"""
        
        if configs is None:
            configs = ['final_optimal', 'balanced_optimal', 'performance_focused']
        
        logger.info(f"开始优化超参数扫描，配置: {configs}")
        
        jobs = []
        balance_strategies = ['moderate', 'focused']
        
        for config in configs:
            for strategy in balance_strategies:
                try:
                    logger.info(f"\n=== 创建配置: {config} + {strategy} ===")
                    job_info = self.continue_training_optimized(
                        prev_job_id=prev_job_id,
                        config_name=config,
                        balance_strategy=strategy,
                        use_best_checkpoint=True
                    )
                    jobs.append(job_info)
                    
                    # 间隔时间避免API限制
                    time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"配置{config}+{strategy}创建失败: {e}")
                    continue
        
        # 保存扫描结果
        sweep_info = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "base_job_id": prev_job_id,
            "configs_tested": configs,
            "balance_strategies": balance_strategies,
            "jobs_created": jobs,
            "total_jobs": len(jobs),
            "optimization_focus": "基于训练分析的参数优化"
        }
        
        with open(f"optimized_hyperparameter_sweep_{sweep_info['timestamp']}.json", 'w') as f:
            json.dump(sweep_info, f, indent=2)
        
        logger.info(f"✅ 优化超参数扫描完成，共创建{len(jobs)}个作业")
        
        return jobs

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        print("用法:")
        print("  python 12_optimized_continue_finetuning.py single <prev_job_id> [config] [strategy]")
        print("  python 12_optimized_continue_finetuning.py sweep <prev_job_id> [config1,config2,...]")
        print("")
        print("配置选项:")
        print("  final_optimal     - 最终优化配置（基于4轮分析，强烈推荐）")
        print("  balanced_optimal  - 平衡最优配置")
        print("  conservative_stable - 保守稳定配置")
        print("  performance_focused - 性能优先配置")
        print("  extended_training - 延长训练配置")
        print("")
        print("平衡策略:")
        print("  gentle   - 温和平衡")
        print("  moderate - 适中平衡（推荐）")
        print("  focused  - 聚焦平衡")
        return
    
    action = sys.argv[1]
    prev_job_id = sys.argv[2]
    
    finetuner = OptimizedContinuedFineTuner()
    
    if action == "single":
        config_name = sys.argv[3] if len(sys.argv) > 3 else 'final_optimal'
        balance_strategy = sys.argv[4] if len(sys.argv) > 4 else 'moderate'
        
        job_info = finetuner.continue_training_optimized(
            prev_job_id=prev_job_id,
            config_name=config_name,
            balance_strategy=balance_strategy,
            use_best_checkpoint=True
        )
        
        print(f"\n✅ 优化继续训练作业创建完成!")
        print(f"作业ID: {job_info['continue_job_id']}")
        print(f"配置: {config_name}")
        print(f"平衡策略: {balance_strategy}")
        
    elif action == "sweep":
        configs = sys.argv[3].split(',') if len(sys.argv) > 3 else None
        jobs = finetuner.run_optimized_sweep(prev_job_id, configs)
        
        print(f"\n✅ 优化超参数扫描完成!")
        print(f"创建了{len(jobs)}个优化作业")
        for job in jobs:
            print(f"  - {job['config_name']}+{job['balance_strategy']}: {job['continue_job_id']}")
    
    else:
        print(f"未知操作: {action}")

if __name__ == "__main__":
    main()