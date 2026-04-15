# Informer 训练结果查看指南

## 📁 结果文件说明

训练完成后，会在 `./results/你的实验名称/` 目录下生成以下文件：

### 1. **metrics.npy** - 评估指标
包含 5 个关键指标：
- **MAE** (Mean Absolute Error) - 平均绝对误差
- **MSE** (Mean Squared Error) - 均方误差
- **RMSE** (Root Mean Squared Error) - 均方根误差
- **MAPE** (Mean Absolute Percentage Error) - 平均绝对百分比误差
- **MSPE** (Mean Squared Percentage Error) - 均方百分比误差

### 2. **pred.npy** - 预测值
- 形状：`(样本数，pred_len, 变量数)`
- 模型在测试集上的预测结果

### 3. **true.npy** - 真实值
- 形状：`(样本数，pred_len, 变量数)`
- 测试集的真实标签

### 4. **real_prediction.npy** - 未来预测（可选）
- 只有在运行了 `--do_predict` 参数时才会生成
- 对未来 unseen 数据的预测结果

---

## 🔧 查看方法

### 方法一：使用 Python 脚本（推荐新手）

我已经创建了 `view_results.py` 脚本：

```bash
python view_results.py
```

这个脚本会：
- 自动查找 results 目录下的所有 .npy 文件
- 显示评估指标
- 显示预测结果的统计信息
- 对比预测值和真实值
- 可选生成可视化图表

### 方法二：使用 Jupyter Notebook（推荐）

我已经创建了 `view_results.ipynb` Notebook：

1. **启动 Jupyter**：
   ```bash
   jupyter notebook view_results.ipynb
   ```

2. **逐步运行各个单元格**：
   - Cell 1: 导入必要的库
   - Cell 2: 列出所有结果文件
   - Cell 3: 查看评估指标（带柱状图）
   - Cell 4: 查看预测结果统计信息
   - Cell 5: 可视化预测曲线 vs 真实曲线
   - Cell 6: 误差分布分析（直方图、箱线图）
   - Cell 7: 查看未来预测结果

3. **修改配置**：
   如果需要查看其他实验的结果，修改 Notebook 中的：
   ```python
   results_dir = './results/re/'  # 改为你的实验目录
   ```

### 方法三：手动编写代码查看

```python
import numpy as np

# 加载文件
metrics = np.load('./results/re/metrics.npy')
preds = np.load('./results/re/pred.npy')
trues = np.load('./results/re/true.npy')

# 查看内容
print("Metrics:", metrics)
print("Predictions shape:", preds.shape)
print("True labels shape:", trues.shape)

# 查看前 5 个预测值
print("First 5 predictions:", preds[:5])
```

---

## 📊 可视化示例

### 查看评估指标柱状图
```python
import matplotlib.pyplot as plt

metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
plt.bar(metric_names, metrics)
plt.show()
```

### 绘制预测曲线对比
```python
plt.figure(figsize=(12, 6))
plt.plot(preds[0, :, 0], 'r-', label='Prediction')
plt.plot(trues[0, :, 0], 'b-', label='True')
plt.legend()
plt.show()
```

---

## 💡 常见问题

### Q1: 文件在哪里？
A: 默认在 `./results/你的实验名称/` 目录下。例如：
- `./results/re/metrics.npy`
- `./results/re/pred.npy`

### Q2: 如何打开 .npy 文件？
A: 使用 `numpy.load()` 函数：
```python
import numpy as np
data = np.load('your_file.npy')
```

### Q3: 数据形状是什么意思？
A: 以 `(32, 24, 7)` 为例：
- `32`: batch size（批次大小）
- `24`: pred_len（预测步长，如预测未来 24 小时）
- `7`: 变量数（如 7 个不同的特征）

### Q4: 如何导出为 Excel 或 CSV？
```python
import pandas as pd

# 加载数据
preds = np.load('./results/re/pred.npy')

# 展平为 2D
preds_2d = preds.reshape(-1, preds.shape[-1])

# 保存为 CSV
pd.DataFrame(preds_2d).to_csv('predictions.csv', index=False)

# 保存为 Excel
pd.DataFrame(preds_2d).to_excel('predictions.xlsx', index=False)
```

### Q5: 如何比较多个实验的结果？
```python
# 加载多个实验的 metrics
metrics_exp1 = np.load('./results/exp1/metrics.npy')
metrics_exp2 = np.load('./results/exp2/metrics.npy')

# 对比 MSE
print(f"Exp1 MSE: {metrics_exp1[1]:.6f}")
print(f"Exp2 MSE: {metrics_exp2[1]:.6f}")
```

---

## 🎯 快速开始

**最简单的方式**：

1. 打开终端或命令提示符
2. 运行：
   ```bash
   jupyter notebook view_results.ipynb
   ```
3. 在浏览器中逐个单元格运行
4. 查看各种统计信息和可视化图表

---

## 📝 注意事项

1. **路径问题**：确保 `results_dir` 指向正确的目录
2. **依赖安装**：需要安装 `numpy`, `matplotlib`, `pandas`（可选）
3. **文件大小**：pred.npy 和 true.npy 可能很大（几 MB 到几百 MB）
4. **内存占用**：如果文件太大，建议分批加载或使用抽样查看

---

## 🚀 进阶用法

### 自定义分析
```python
# 计算特定时间步的误差
time_step = 12  # 查看第 12 个时间步
error_at_step = np.abs(preds[:, time_step, :] - trues[:, time_step, :])
print(f"Error at step {time_step}: {error_at_step.mean():.6f}")

# 分析不同变量的表现
for var_idx in range(preds.shape[2]):
    mae = np.abs(preds[:, :, var_idx] - trues[:, :, var_idx]).mean()
    print(f"Variable {var_idx} MAE: {mae:.6f}")
```

### 批量处理多个实验
```python
import glob

# 查找所有实验目录
exp_dirs = glob.glob('./results/*/')

for exp_dir in exp_dirs:
    metrics_file = os.path.join(exp_dir, 'metrics.npy')
    if os.path.exists(metrics_file):
        metrics = np.load(metrics_file)
        print(f"{exp_dir}: MSE={metrics[1]:.6f}")
```

---

祝你分析顺利！如有问题欢迎随时询问。 😊
