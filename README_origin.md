# Informer 项目说明

本项目是 [Informer](https://arxiv.org/abs/2012.07436) 模型的 PyTorch 实现，用于长序列时间序列预测任务。

## 目录
- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [训练与测试](#训练与测试)
- [预测与可视化](#预测与可视化)
- [项目结构](#项目结构)
- [参数说明](#参数说明)
- [引用](#引用)

## 环境要求

- Python 3.6+
- PyTorch 1.8.0+
- matplotlib
- numpy
- pandas
- scikit-learn

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

### ETT 数据集
ETT 数据集可以从 [ETDataset](https://github.com/zhouhaoyi/ETDataset) 下载，需要将数据文件放入 `data/ETT/` 文件夹中。

### 其他数据集
ECL 和 Weather 数据集可以从以下链接下载：
- [Google Drive](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing)
- [BaiduPan](https://pan.baidu.com/s/1wyaGUisUICYHnfkZzWCwyA), password: 6gan

数据文件格式应为 CSV，包含日期列（`date`）和数值列。

## 快速开始

### 训练模型
```bash
# ETTh1 数据集，多变量预测多变量
python -u main_informer.py --model informer --data ETTh1 --features M --attn prob

# ETTh1 数据集，单变量预测单变量
python -u main_informer.py --model informer --data ETTh1 --features S --attn prob
```

### 测试模型
```bash
python -u main_informer.py --model informer --data ETTh1 --features M --attn prob --use_gpu False --itr 1
```

## 训练与测试

### 不同数据集的训练命令

#### ETTh1
```bash
# 多变量预测
python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

# 单变量预测
python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_len 168 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5
```

#### ETTm1（分钟级数据）
```bash
python -u main_informer.py --model informer --data ETTm1 --features M --attn prob --freq t
```

### 预设脚本
项目提供了预设的训练脚本在 `scripts/` 目录下：
```bash
bash scripts/ETTh1.sh
```

## 预测与可视化

### 单次预测
```bash
python -u main_informer.py --model informer --data ETTh1 --features M --do_predict --attn prob
```

### 结果可视化
```bash
# 可视化时间序列
python -u informer_visualize_sequence.py

# 可视化预测结果
python -u informer_visualize_forecast.py
```

### 查看结果
```bash
python -u view_results.py
```

## 项目结构

```
Informer2020-main/
├── checkpoints/          # 模型检查点（训练生成）
├── data/                 # 数据文件
│   ├── ETT/             # ETT 数据集
│   └── PJME_hourly.csv  # PJME 电力数据
├── exp/                 # 实验相关代码
│   ├── exp_basic.py
│   └── exp_informer.py
├── models/              # 模型架构
│   ├── attn.py         # 注意力机制
│   ├── decoder.py      # 解码器
│   ├── embed.py        # 嵌入层
│   ├── encoder.py      # 编码器
│   └── model.py        # 主模型
├── results/             # 结果输出（预测生成）
├── scripts/             # 训练脚本
│   ├── ETTh1.sh
│   ├── ETTh2.sh
│   ├── ETTm1.sh
│   └── WTH.sh
├── utils/               # 工具函数
│   ├── masking.py      # 掩码操作
│   ├── metrics.py      # 评估指标
│   ├── timefeatures.py # 时间特征编码
│   └── tools.py        # 工具函数
├── main_informer.py     # 主训练程序
├── informer_visualize_forecast.py  # 预测可视化
├── informer_visualize_sequence.py  # 序列可视化
├── view_results.py      # 结果查看
└── requirements.txt     # 依赖列表
```

## 参数说明

### 必选参数
- `--model`: 模型类型（`informer`, `informerstack`, `informerlight`）
- `--data`: 数据集名称（`ETTh1`, `ETTh2`, `ETTm1`, `WTH` 等）

### 主要参数
- `--features`: 预测任务类型
  - `M`: 多变量预测多变量
  - `S`: 单变量预测单变量
  - `MS`: 多变量预测单变量
- `--target`: 目标特征（默认：`OT`）
- `--freq`: 时间特征编码频率（`s`, `t`, `h`, `d`, `b`, `w`, `m`）
- `--seq_len`: 输入序列长度（默认：168）
- `--label_len`: 解码器起始 token 长度（默认：48）
- `--pred_len`: 预测序列长度（默认：24）

### 模型参数
- `--d_model`: 模型维度（默认：512）
- `--n_heads`: 注意力头数量（默认：8）
- `--e_layers`: 编码器层数（默认：2）
- `--d_layers`: 解码器层数（默认：1）
- `--d_ff`: 前馈网络维度（默认：2048）
- `--attn`: 注意力类型（`prob`, `full`）

### 训练参数
- `--train_epochs`: 训练轮数（默认：10）
- `--batch_size`: 批大小（默认：32）
- `--learning_rate`: 学习率（默认：0.0001）
- `--itr`: 实验次数（默认：2）

### 其他参数
- `--checkpoints`: 模型检查点路径（默认：`./checkpoints/`）
- `--use_gpu`: 是否使用 GPU（默认：True）
- `--gpu`: GPU 编号（默认：0）
- `--des`: 实验描述（默认：`test`）

## 评估指标

模型输出以下指标：
- **MSE**: 均方误差
- **MAE**: 平均绝对误差

结果保存在 `results/` 目录下。

## 常见问题

### 1. 数据标准化
项目默认对数据进行零均值、单位方差的标准化处理，通过 `scale` 参数控制。

### 2. 时间特征编码
支持两种编码方式：
- `timeenc=0`: 原始数值编码（月、日、小时等）
- `timeenc=1`: 归一化编码（[-0.5, 0.5]）

通过 `--embed timeF` 使用时间特征编码。

### 3. 内存不足
- 减小 `--batch_size`
- 减小 `--seq_len`
- 使用更小的模型（减少 `--d_model`、`--n_heads`）

### 4. torch 版本兼容性
如果遇到 `Conv1d` 相关错误，检查 PyTorch 版本，可能需要修改 `models/embed.py` 中的 `TokenEmbedding` 类。

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

## 许可证

本项目采用 [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。

## 联系方式

如有问题，请通过以下方式联系：
- Email: zhouhaoyi1991@gmail.com
- GitHub Issues
