#!/usr/bin/env python3
"""
将 personas_for_prediction.json 转换为 parquet 格式
提高读取效率，减少文件大小
"""

import json
import pandas as pd
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_personas_to_parquet():
    """将 personas_for_prediction.json 转换为 parquet 格式"""
    
    data_dir = Path("data")
    json_file = data_dir / "personas_for_prediction.json"
    parquet_file = data_dir / "personas_for_prediction.parquet"
    
    logger.info("开始转换 personas_for_prediction.json 到 parquet 格式...")
    
    # 检查输入文件
    if not json_file.exists():
        logger.error(f"❌ 输入文件不存在: {json_file}")
        return False
    
    try:
        # 读取JSON文件
        logger.info(f"读取JSON文件: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            personas_dict = json.load(f)
        
        logger.info(f"JSON数据: {len(personas_dict)} 个personas")
        
        # 转换为DataFrame
        logger.info("转换为DataFrame格式...")
        rows = []
        for persona_id, persona_text in personas_dict.items():
            # 提取pid
            if persona_id.startswith('pid_'):
                pid = int(persona_id.split('_')[1])
            else:
                logger.warning(f"无法解析persona_id: {persona_id}")
                continue
                
            rows.append({
                'pid': pid,
                'persona_id': persona_id,
                'persona_text': persona_text
            })
        
        df = pd.DataFrame(rows)
        
        # 按pid排序
        df = df.sort_values('pid').reset_index(drop=True)
        
        logger.info(f"DataFrame: {len(df)} 行, 列: {list(df.columns)}")
        
        # 保存为parquet
        logger.info(f"保存为parquet格式: {parquet_file}")
        df.to_parquet(parquet_file, index=False)
        
        # 检查文件大小
        json_size_mb = json_file.stat().st_size / (1024 * 1024)
        parquet_size_mb = parquet_file.stat().st_size / (1024 * 1024)
        
        logger.info(f"文件大小对比:")
        logger.info(f"  JSON: {json_size_mb:.2f} MB")
        logger.info(f"  Parquet: {parquet_size_mb:.2f} MB")
        logger.info(f"  压缩率: {(1 - parquet_size_mb/json_size_mb)*100:.1f}%")
        
        # 验证数据
        logger.info("验证转换后的数据...")
        test_df = pd.read_parquet(parquet_file)
        
        if len(test_df) == len(df) and test_df['pid'].nunique() == len(personas_dict):
            logger.info("✅ 数据验证通过")
            
            # 显示样本
            logger.info(f"数据样本 (前3行):")
            for i in range(min(3, len(test_df))):
                row = test_df.iloc[i]
                text_preview = row['persona_text'][:100] + "..." if len(row['persona_text']) > 100 else row['persona_text']
                logger.info(f"  PID {row['pid']}: {text_preview}")
            
            logger.info(f"✅ 转换成功完成!")
            return True
        else:
            logger.error("❌ 数据验证失败")
            return False
        
    except Exception as e:
        logger.error(f"❌ 转换失败: {e}")
        return False

def load_personas_from_parquet():
    """从parquet文件加载personas数据（演示用法）"""
    
    data_dir = Path("data")
    parquet_file = data_dir / "personas_for_prediction.parquet"
    
    if not parquet_file.exists():
        logger.error(f"❌ Parquet文件不存在: {parquet_file}")
        return None
    
    try:
        logger.info("从parquet文件加载personas...")
        df = pd.read_parquet(parquet_file)
        
        # 转换回字典格式（与原JSON相同）
        personas_dict = {}
        for _, row in df.iterrows():
            personas_dict[row['persona_id']] = row['persona_text']
        
        logger.info(f"✅ 成功加载 {len(personas_dict)} 个personas")
        return personas_dict
        
    except Exception as e:
        logger.error(f"❌ 加载失败: {e}")
        return None

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-load":
        # 测试加载parquet文件
        personas = load_personas_from_parquet()
        if personas:
            print(f"✅ 成功从parquet加载 {len(personas)} 个personas")
            # 显示第一个样本
            first_key = list(personas.keys())[0]
            first_text = personas[first_key][:200] + "..." if len(personas[first_key]) > 200 else personas[first_key]
            print(f"样本 {first_key}: {first_text}")
        else:
            print("❌ 加载失败")
        return
    
    # 执行转换
    logger.info("🚀 开始转换personas数据到parquet格式...")
    success = convert_personas_to_parquet()
    
    if success:
        print("\n" + "="*60)
        print("🎉 Personas数据转换完成!")
        print("📁 输出文件: data/personas_for_prediction.parquet")
        print("💡 使用方法:")
        print("   import pandas as pd")
        print("   df = pd.read_parquet('data/personas_for_prediction.parquet')")
        print("="*60)
        print("\n测试加载: python convert_personas_to_parquet.py --test-load")
    else:
        print("❌ 转换失败")

if __name__ == "__main__":
    main()