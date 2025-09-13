#!/usr/bin/env python3
"""
导出与 2058 个受试者一一对应的 persona 数据
基于提供的参考代码，保存到 data/ 目录
"""

from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_full_personas():
    """提取完整的persona数据并保存到data目录"""
    
    # 确保data目录存在
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("从Hugging Face加载full_persona数据...")
    
    try:
        # 载入"权威 persona"配置（与所有 pid 一一对应）
        ds = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona")["data"]
        df = ds.to_pandas()
        
        # 基本完整性校验
        n_total = len(df)
        n_pid = df["pid"].nunique()
        logger.info(f"总数据: {n_total} 行, 唯一pid: {n_pid}")
        
        if n_total != n_pid:
            logger.warning("⚠️ 存在重复 pid，请检查上游数据")
        
        if n_total != 2058:
            logger.info(f"ℹ️ 当前样本量为 {n_total}（不是2058），以实际值为准")
        
        # 导出三种常用形式
        logger.info("导出persona数据到data目录...")
        
        # 1. Persona Summary (最常用于LLM预测)
        summary_df = df[["pid", "persona_summary"]].copy()
        summary_df.to_csv(data_dir / "persona_summary.csv", index=False)
        logger.info(f"✅ 保存 persona_summary.csv: {len(summary_df)} 条记录")
        
        # 2. Persona Text (详细文本)
        text_df = df[["pid", "persona_text"]].copy()
        text_df.to_csv(data_dir / "persona_text.csv", index=False)
        logger.info(f"✅ 保存 persona_text.csv: {len(text_df)} 条记录")
        
        # 3. Persona JSON (结构化数据)
        json_df = df[["pid", "persona_json"]].copy()
        json_df.to_csv(data_dir / "persona_json.csv", index=False)
        logger.info(f"✅ 保存 persona_json.csv: {len(json_df)} 条记录")
        
        # 4. 完整数据 (parquet格式，高效读取)
        df.to_parquet(data_dir / "full_personas.parquet", index=False)
        logger.info(f"✅ 保存 full_personas.parquet: 完整数据")
        
        # 5. 为预测系统准备的personas字典 (JSON格式)
        personas_dict = {}
        for _, row in df.iterrows():
            pid = row['pid']
            # 优先使用 persona_summary，回退到 persona_text
            persona_text = row.get('persona_summary') or row.get('persona_text', '')
            personas_dict[f"pid_{pid}"] = persona_text
        
        with open(data_dir / "personas_for_prediction.json", 'w', encoding='utf-8') as f:
            json.dump(personas_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 保存 personas_for_prediction.json: {len(personas_dict)} 个personas")
        
        # 6. 验证文件
        logger.info("\n📋 文件验证:")
        for file_path in [
            "persona_summary.csv",
            "persona_text.csv", 
            "persona_json.csv",
            "full_personas.parquet",
            "personas_for_prediction.json"
        ]:
            full_path = data_dir / file_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path}: {size_mb:.2f} MB")
            else:
                logger.warning(f"  ❌ {file_path}: 文件未生成")
        
        # 7. 显示数据预览
        logger.info(f"\n📊 数据预览 (前3个personas):")
        for i in range(min(3, len(df))):
            pid = df.iloc[i]['pid']
            summary = df.iloc[i].get('persona_summary', '')[:100] + "..."
            logger.info(f"  PID {pid}: {summary}")
        
        logger.info(f"\n🎉 Persona数据提取完成!")
        logger.info(f"总计: {len(df)} 个personas")
        logger.info(f"保存目录: {data_dir.absolute()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 提取persona数据失败: {e}")
        return False

def verify_personas_data():
    """验证已保存的persona数据"""
    
    data_dir = Path("data")
    
    logger.info("验证保存的persona数据...")
    
    # 检查关键文件
    required_files = [
        "persona_summary.csv",
        "personas_for_prediction.json"
    ]
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            logger.error(f"❌ 缺少文件: {file_name}")
            return False
        else:
            logger.info(f"✅ 找到文件: {file_name}")
    
    # 验证数据内容
    try:
        # 检查CSV
        summary_df = pd.read_csv(data_dir / "persona_summary.csv")
        logger.info(f"persona_summary.csv: {len(summary_df)} 行, 列: {list(summary_df.columns)}")
        
        # 检查JSON
        with open(data_dir / "personas_for_prediction.json", 'r', encoding='utf-8') as f:
            personas_dict = json.load(f)
        logger.info(f"personas_for_prediction.json: {len(personas_dict)} 个personas")
        
        # 检查pid对应关系
        csv_pids = set(summary_df['pid'])
        json_pids = set(int(k.split('_')[1]) for k in personas_dict.keys() if k.startswith('pid_'))
        
        if csv_pids == json_pids:
            logger.info("✅ CSV和JSON中的PID完全对应")
        else:
            logger.warning(f"⚠️ PID不完全对应: CSV={len(csv_pids)}, JSON={len(json_pids)}")
        
        logger.info("✅ 数据验证通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据验证失败: {e}")
        return False

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # 仅验证现有数据
        success = verify_personas_data()
        if success:
            print("✅ Persona数据验证通过")
        else:
            print("❌ Persona数据验证失败")
        return
    
    # 提取并保存数据
    logger.info("🚀 开始提取persona数据...")
    success = extract_full_personas()
    
    if success:
        # 验证保存的数据
        logger.info("\n🔍 验证保存的数据...")
        verify_success = verify_personas_data()
        
        if verify_success:
            print("\n" + "="*60)
            print("🎉 Persona数据提取和验证完成!")
            print("📁 数据保存位置: ./data/")
            print("📝 主要文件:")
            print("  - persona_summary.csv (用于LLM预测)")
            print("  - personas_for_prediction.json (预测系统格式)")
            print("  - full_personas.parquet (完整数据)")
            print("="*60)
            print("\n下一步: 使用这些persona数据进行预测验证")
        else:
            print("❌ 数据验证失败，请检查文件")
    else:
        print("❌ Persona数据提取失败")

if __name__ == "__main__":
    main()