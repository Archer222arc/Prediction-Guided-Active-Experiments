# 全量Ground Truth预测指南

## 🎯 处理全部2058个personas

现在您可以对整个数字孪生数据集的所有2058个personas生成预测结果。

## 🚀 调用方法

### **1. 全量处理（推荐配置）**

```bash
# 处理全部2058个personas，使用base和cot两种方法
python predict_ground_truth.py both -1 15 3

# 参数说明:
# both: 运行base和cot两种方法
# -1: 处理全部2058个personas
# 15: 使用15个并发线程（较高并发）
# 3: 每次API调用最多重试3次
```

### **2. 仅单一方法**

```bash
# 仅base方法
python predict_ground_truth.py base -1 12 5

# 仅cot方法  
python predict_ground_truth.py cot -1 12 5
```

### **3. 保守配置（网络不稳定时）**

```bash
# 使用较低并发和更多重试
python predict_ground_truth.py both -1 8 8
```

## 📊 性能预期

### **资源消耗**
- **API调用总数**: 20,580次 (2058 personas × 10 questions)
- **预计时间**: 2-4小时（取决于并发数和网络状况）
- **内存使用**: ~100MB
- **输出文件大小**: ~1-2MB CSV文件

### **时间估算表**

| 并发线程 | 预计时间 | 适用场景 |
|----------|----------|----------|
| 8线程 | 4-5小时 | 网络不稳定 |
| 12线程 | 3-4小时 | 标准网络 |
| 15线程 | 2-3小时 | 良好网络 |
| 20线程 | 2小时 | 优秀网络（可能触发限制）|

## 📝 输出文件

### **文件命名**
```
ground_truth_predictions_FULL_base_20250913_143022.csv
ground_truth_predictions_FULL_cot_20250913_174536.csv
```

### **文件内容**
每个CSV文件包含：
- **persona_id**: 数字人ID (gt_pid_1, gt_pid_2, ...)
- **10个政策问题的预测结果**: carbon_tax_LLM, clean_energy_LLM, ...
- **值范围**: 1-5 (1=强烈反对, 5=强烈支持)

## 🔍 进度监控

脚本会自动提供详细的进度报告：

```
2025-09-13 14:30:22 - INFO - 实际加载 2058 个personas，预计 20580 次API调用
2025-09-13 14:30:22 - INFO - 🚀 大规模处理开始，建议监控进度...
2025-09-13 14:30:22 - INFO - 预估完成时间: 137 分钟
...
2025-09-13 17:45:30 - INFO - 📊 处理统计:
2025-09-13 17:45:30 - INFO -   - 成功处理: 2053/2058 (99.8%)
2025-09-13 17:45:30 - INFO -   - 平均每个persona: 0.09 分钟
2025-09-13 17:45:30 - INFO -   - API调用效率: 106.3 次/分钟
```

## ⚠️ 注意事项

### **运行前准备**
1. **确保网络稳定**: 过程需要2-4小时
2. **检查磁盘空间**: 确保有足够空间存储结果
3. **选择合适时间**: 建议在网络负载较低时运行

### **错误处理**
- 脚本有完整的重试机制
- 失败的API调用会自动重试
- 最终失败的项目会标记为None

### **中断恢复**
- 如果需要中断，可以用Ctrl+C
- 脚本会保存已完成的部分
- 可以通过调整参数重新运行未完成的部分

## 🎯 实际使用建议

### **首次运行**
```bash
# 先用小量数据测试
python predict_ground_truth.py both 50 8 5

# 确认无问题后运行全量
python predict_ground_truth.py both -1 15 3
```

### **生产环境**
```bash
# 推荐配置
nohup python predict_ground_truth.py both -1 15 3 > full_prediction.log 2>&1 &

# 查看进度
tail -f full_prediction.log
```

这样您就可以获得完整的2058个数字孪生personas在10个政策议题上的预测结果，为后续的分析和研究提供完整的数据基础。