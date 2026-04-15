# PJME 电网负荷预测项目

本项目使用 Informer 模型对 PJME 电网负荷数据进行时间序列预测。

## 数据集介绍

### PJME 电力负荷数据
- **数据来源**: PJM Interconnection LLC（美国电力公司）
- **时间范围**: 2002年12月31日 至 2018年6月
- **频率**: 小时级（hourly）
- **样本数量**: 约 146,096 小时
- **目标变量**: `PJME_MW` - 电力负荷（兆瓦）
- **数据特点**: 
  - 单变量时间序列
  - 具有明显的日周期性和年周期性
  - 负荷值范围：约 20,000 - 45,000 MW
  - 受天气、节假日等因素影响

### 数据格式
```csv
Datetime,PJME_MW
2002-12-31 01:00:00,26498.0
2002-12-31 02:00:00,25147.0
2002-12-31 03:00:00,24574.0
```

## 项目结构

```
Informer2020-main/
├── data/
│   └── PJME_hourly.csv          # PJME 电力负荷数据
├── checkpoints/                   # 模型检查点
│   └── informer_PJME_*/          # PJME 模型训练的检查点
├── results/                       # 预测结果
│   └── informer_PJME_*/          # 预测结果和可视化
│       ├── pred.npy              # 预测值
│       ├── true.npy              # 真实值
│       ├── metrics.npy           # 评估指标
│       └── pred_plot.png         # 预测结果图
├── models/                        # 模型架构
├── exp/                           # 实验相关代码
├── utils/                         # 工具函数
├── main_informer.py              # 主训练程序
├── informer_visualize_forecast.py # 预测可视化
└── requirements.txt              # 依赖列表
```

## 环境要求

- Python 3.6+
- PyTorch 1.8.0+
- numpy
- pandas
- matplotlib
- scikit-learn

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
# 基础训练（单变量预测）
python -u main_informer.py \
    --model informer \
    --data PJME \
    --features S \
    --root_path ./data/ \
    --data_path PJME_hourly.csv \
    --target PJME_MW \
    --freq h \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --des 'PJME_exp' \
    --itr 1
```

### 2. 测试模型

```bash
python -u main_informer.py \
    --model informer \
    --data PJME \
    --features S \
    --root_path ./data/ \
    --data_path PJME_hourly.csv \
    --target PJME_MW \
    --freq h \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --use_gpu False \
    --itr 1
```

### 3. 单次预测

```bash
python -u main_informer.py \
    --model informer \
    --data PJME \
    --features S \
    --root_path ./data/ \
    --data_path PJME_hourly.csv \
    --target PJME_MW \
    --freq h \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --attn prob \
    --do_predict \
    --use_gpu False
```

### 4. 结果可视化

```bash
# 可视化预测结果
python -u informer_visualize_forecast.py
```

## 参数说明

### 必选参数
- `--model`: 模型类型（`informer`）
- `--data`: 数据集名称（`PJME`）
- `--root_path`: 数据根路径（`./data/`）
- `--data_path`: 数据文件名（`PJME_hourly.csv`）
- `--target`: 目标变量（`PJME_MW`）

### 主要参数
- `--features`: 预测任务类型
  - `S`: 单变量预测单变量（PJME 使用此模式）
- `--freq`: 时间特征编码频率（`h` - 小时级）
- `--seq_len`: 输入序列长度（默认：96，约4天）
- `--label_len`: 解码器起始 token 长度（默认：48）
- `--pred_len`: 预测序列长度（默认：24）

### 模型参数
- `--d_model`: 模型维度（默认：512）
- `--n_heads`: 注意力头数量（默认：8）
- `--e_layers`: 编码器层数（默认：2）
- `--d_layers`: 解码器层数（默认：1）
- `--d_ff`: 前馈网络维度（默认：2048）
- `--attn`: 注意力类型（`prob` - ProbSparse）

### 训练参数
- `--train_epochs`: 训练轮数（默认：10）
- `--batch_size`: 批大小（默认：32）
- `--learning_rate`: 学习率（默认：0.0001）
- `--itr`: 实验次数（默认：2）

### 其他参数
- `--checkpoints`: 模型检查点路径（默认：`./checkpoints/`）
- `--use_gpu`: 是否使用 GPU（默认：True）
- `--gpu`: GPU 编号（默认：0）

## 实验配置

### 推荐配置（PJME 数据集）

| 参数 | 值 | 说明 |
|------|-----|------|
| seq_len | 96 | 4天的历史数据 |
| label_len | 48 | 解码器起始长度 |
| pred_len | 24 | 预测未来24小时 |
| e_layers | 2 | 编码器2层 |
| d_layers | 1 | 解码器1层 |
| d_model | 512 | 模型维度 |
| n_heads | 8 | 注意力头数 |
| batch_size | 32 | 批大小 |
| learning_rate | 0.0001 | 学习率 |

### 不同预测长度的配置

```bash
# 短期预测（24小时）
python -u main_informer.py --model informer --data PJME --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob

# 中期预测（48小时）
python -u main_informer.py --model informer --data PJME --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob

# 长期预测（168小时）
python -u main_informer.py --model informer --data PJME --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob
```

## 评估指标

模型输出以下评估指标：

### 主要指标
- **MSE (Mean Squared Error)**: 均方误差
- **MAE (Mean Absolute Error)**: 平均绝对误差

### 结果文件
- `pred.npy`: 预测值数组，形状为 `(n_samples, pred_len, 1)`
- `true.npy`: 真实值数组，形状为 `(n_samples, pred_len, 1)`
- `metrics.npy`: 评估指标数组
- `pred_plot.png`: 预测结果可视化图

## 数据预处理

### 标准化
项目默认对数据进行零均值、单位方差的标准化处理：
```
x_normalized = (x - mean) / std
```

### 时间特征编码
使用 `timeenc=1` 的归一化编码方式，将时间特征编码到 `[-0.5, 0.5]` 区间：
- 小时：`hour / 23.0 - 0.5`
- 星期：`dayofweek / 6.0 - 0.5`
- 月份：`month / 11.0 - 0.5`
- 等等...

### 数据划分
- 训练集：70%
- 验证集：15%
- 测试集：15%

## 可视化示例

### 预测结果图
可视化显示：
- 黑色线：历史真实值
- 红色线：预测值
- 蓝色线：未来真实值（如果可用）

### 模型架构
Informer 使用 ProbSparse 注意力机制，相比标准 Transformer：
- 时间复杂度：O(L log L) vs O(L²)
- 空间复杂度：显著降低
- 适合长序列预测

## 常见问题

### 1. 数据标准化
PJME 数据集默认进行零均值、单位方差标准化。如果需要原始尺度的结果，使用 `--inverse` 参数。

### 2. 时间特征编码
PJME 是小时级数据，使用 `--freq h`，时间特征包括：
- 小时（Hour of day）
- 星期（Day of week）
- 月份（Day of month）
- 年份（Day of year）

### 3. GPU 内存不足
```bash
# 减小批大小
--batch_size 16

# 减小序列长度
--seq_len 48

# 使用 CPU
--use_gpu False
```

### 4. 训练时间过长
```bash
# 减少训练轮数
--train_epochs 5

# 减小模型规模
--d_model 256
--n_heads 4
```

### 5. 预测效果不佳
- 增加训练轮数：`--train_epochs 20`
- 调整学习率：`--learning_rate 0.00005`
- 增加序列长度：`--seq_len 192`
- 使用更长的预测长度：`--pred_len 48`

## 性能优化建议

### 1. 批大小调整
根据 GPU 内存调整：
- GPU 内存 > 8GB: `--batch_size 64`
- GPU 内存 4-8GB: `--batch_size 32`
- GPU 内存 < 4GB: `--batch_size 16`

### 2. 序列长度选择
- 短期预测（1-3天）：`seq_len = 48-96`
- 中期预测（3-7天）：`seq_len = 168-336`
- 长期预测（7天以上）：`seq_len = 720+`

### 3. 学习率调度
项目支持多种学习率调整策略：
- `--lradj type1`: 每个 epoch 后学习率减半
- `--lradj type2`: 预设学习率表

## 引用

如果本项目对您的研究有帮助，请引用以下论文：

```bibtex
@article{haoyietal-informerEx-2023,
  author    = {Haoyi Zhou and
               Jianxin Li and
               Shanghang Zhang and
               Shuai Zhang and
               Mengyi Yan and
               Hui Xiong},
  title     = {Expanding the prediction capacity in long sequence time-series forecasting},
  journal   = {Artificial Intelligence},
  volume    = {318},
  pages     = {103886},
  issn      = {0004-3702},
  year      = {2023},
}
```

```bibtex
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  year      = {2021},
}
```

## 数据集引用

PJME 电力负荷数据来自：
- **来源**: PJM Interconnection LLC
- **访问**: https://www.pjm.com/
- **数据**: PJM Hourly Load Data

## 许可证

本项目采用 [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。

## 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues
- Email: zhouhaoyi1991@gmail.com
