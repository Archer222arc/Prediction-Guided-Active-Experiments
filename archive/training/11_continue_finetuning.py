#!/usr/bin/env python3
"""
继续微调脚本 - 从最佳检查点继续训练，解决系统性偏差
"""

import os
import time
import json
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuedFineTuner:
    """继续微调器 - 基于检查点继续训练"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        azure_config = config["azure_endpoints"]["north_central_us"]
        
        self.client = OpenAI(
            api_key=azure_config["api_key"],
            base_url=azure_config["endpoint"].rstrip("/") + "/openai/v1/"
        )
        
        logger.info("继续微调器初始化完成")
    
    def find_best_checkpoint(self, prev_job_id: str):
        """找到最佳检查点"""
        
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
            best_metric = None
            
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
                        # 尝试作为字典访问
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
    
    def create_balanced_training_data(self, base_file: str, output_file: str):
        """创建平衡的训练数据 - 专门针对UNITY和GAMBLERESTR"""
        
        logger.info("创建平衡的继续训练数据...")
        
        with open(base_file, 'r') as f:
            lines = f.readlines()
        
        balanced_lines = []
        
        # 重权重策略
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
                        # 温和增加UNITY选项1的权重
                        if 'UNITY: 1' in assistant_msg:
                            repeat_count = 2  # 降低到2倍权重
                        
                        # 温和增加GAMBLERESTR选项1和3的权重
                        elif 'GAMBLERESTR: 1' in assistant_msg:
                            repeat_count = 2  # 降低到2倍权重
                        elif 'GAMBLERESTR: 3' in assistant_msg:
                            repeat_count = 2  # 降低到2倍权重
                
                for _ in range(repeat_count):
                    balanced_lines.append(line)
                    
            except:
                balanced_lines.append(line)
        
        # 随机打乱
        import random
        random.shuffle(balanced_lines)
        
        with open(output_file, 'w') as f:
            f.writelines(balanced_lines)
        
        logger.info(f"平衡数据创建完成:")
        logger.info(f"  原始: {len(lines)} 样本")
        logger.info(f"  平衡后: {len(balanced_lines)} 样本")
        logger.info(f"  增加倍率: {len(balanced_lines)/len(lines):.2f}x")
        
        return output_file
    
    def regenerate_validation_set(self, base_train_file: str, output_validation_file: str, validation_ratio: float = 0.15):
        """重新生成更大的验证集"""
        
        logger.info(f"重新生成验证集，比例: {validation_ratio:.1%}...")
        
        with open(base_train_file, 'r') as f:
            all_lines = f.readlines()
        
        # 随机打乱
        import random
        random.shuffle(all_lines)
        
        # 分割数据
        val_size = int(len(all_lines) * validation_ratio)
        val_lines = all_lines[:val_size]
        remaining_train_lines = all_lines[val_size:]
        
        # 保存验证集
        with open(output_validation_file, 'w') as f:
            f.writelines(val_lines)
        
        # 更新训练集
        with open(base_train_file, 'w') as f:
            f.writelines(remaining_train_lines)
        
        logger.info(f"验证集重新生成完成:")
        logger.info(f"  训练样本: {len(remaining_train_lines)}")
        logger.info(f"  验证样本: {len(val_lines)}")
        logger.info(f"  验证比例: {len(val_lines)/(len(remaining_train_lines)+len(val_lines)):.1%}")
        
        return output_validation_file
    
    def continue_training(self, prev_job_id: str, use_best_checkpoint: bool = True):
        """继续训练"""
        
        logger.info(f"开始从作业 {prev_job_id} 继续训练...")
        
        # 确定基础模型
        if use_best_checkpoint:
            base_model, best_score, best_step = self.find_best_checkpoint(prev_job_id)
            suffix = f"cont-ckpt-step{best_step}"
        else:
            job = self.client.fine_tuning.jobs.retrieve(prev_job_id)
            base_model = job.fine_tuned_model
            suffix = "cont-final"
        
        logger.info(f"基础模型: {base_model}")
        
        # 创建平衡的训练数据
        balanced_train_file = self.create_balanced_training_data(
            "finetuning_data/train.jsonl",
            "finetuning_data/train_balanced_continue.jsonl"
        )
        
        # 重新生成验证集
        validation_file = self.regenerate_validation_set(
            balanced_train_file,
            "finetuning_data/validation_regenerated.jsonl",
            validation_ratio=0.15
        )
        
        # 上传文件
        logger.info("上传平衡训练文件...")
        tr = self.client.files.create(
            file=open(balanced_train_file, "rb"), 
            purpose="fine-tune"
        )
        
        logger.info("上传重新生成的验证文件...")
        vr = self.client.files.create(
            file=open(validation_file, "rb"), 
            purpose="fine-tune"
        )
        
        # 等待文件处理
        self._wait_for_file_processing(tr.id)
        self._wait_for_file_processing(vr.id)
        
        # 修复后的超参数 - 解决训练不稳定问题
        hyperparams = {
            "batch_size": 64,                    # 提高batch size增强稳定性
            "learning_rate_multiplier": 0.5,     # 降低学习率避免震荡
            "n_epochs": 1,                       # 保持1个epoch
            "seed": 42
        }
        
        logger.info("创建继续训练作业...")
        logger.info(f"超参数: {hyperparams}")
        
        job = self.client.fine_tuning.jobs.create(
            model=base_model,               # 使用检查点或最终模型
            training_file=tr.id,
            validation_file=vr.id,
            hyperparameters=hyperparams,
            suffix=suffix
        )
        
        # 保存作业信息
        job_info = {
            "continue_job_id": job.id,
            "base_job_id": prev_job_id,
            "base_model": base_model,
            "use_best_checkpoint": use_best_checkpoint,
            "hyperparameters": hyperparams,
            "created_at": job.created_at,
            "status": job.status,
            "training_samples": len(open(balanced_train_file).readlines())
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_file = f"continue_finetuning_job_{timestamp}.json"
        
        with open(job_file, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        logger.info(f"✅ 继续训练作业创建成功!")
        logger.info(f"新作业ID: {job.id}")
        logger.info(f"基于: {base_model}")
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

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 11_continue_finetuning.py <prev_job_id> [checkpoint|final]")
        print("")
        print("示例:")
        print("  python 11_continue_finetuning.py ftjob-xxx checkpoint  # 从最佳检查点继续(推荐)")
        print("  python 11_continue_finetuning.py ftjob-xxx final      # 从最终模型继续")
        return
    
    prev_job_id = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "checkpoint"
    
    use_checkpoint = (method == "checkpoint")
    
    finetuner = ContinuedFineTuner()
    
    job_info = finetuner.continue_training(
        prev_job_id=prev_job_id,
        use_best_checkpoint=use_checkpoint
    )
    
    print(f"\n✅ 继续训练已启动!")
    print(f"新作业ID: {job_info['continue_job_id']}")
    print(f"基于: {job_info['base_model']}")
    print(f"方法: {'最佳检查点' if use_checkpoint else '最终模型'}")

if __name__ == "__main__":
    main()