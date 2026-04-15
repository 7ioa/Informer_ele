import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.model import Informer
from data.data_loader import Dataset_Custom


SEQ_LEN = 168
LABEL_LEN = 48  # 训练时使用的 label_len，用于构造解码器输入
PRED_LEN = 24


def build_model(device: torch.device):
    """
    按照项目中默认的 Informer 配置构建模型。
    这里的超参数需与你训练 PJME 时保持一致。
    """
    model = Informer(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        attn='prob',
        embed='timeF',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        device=device,
    ).float()
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 文件不存在：{ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_test_dataset(data_root: str, data_path: str, target: str, features: str):
    """
    与训练时 Dataset_Custom 的使用保持一致，只是这里直接构造 test 集。
    """
    dataset = Dataset_Custom(
        root_path=data_root,
        data_path=data_path,
        flag='test',
        size=[SEQ_LEN, LABEL_LEN, PRED_LEN],
        features=features,
        target=target,
        timeenc=1,
        freq='h',
    )
    return dataset


def select_sample(dataset: Dataset_Custom, sample_idx: int):
    n_samples = len(dataset)
    if sample_idx < 0 or sample_idx >= n_samples:
        raise IndexError(f"sample_idx 超出范围：0 ~ {n_samples - 1}")

    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[sample_idx]

    # history: [SEQ_LEN]
    if isinstance(seq_x, np.ndarray):
        history = seq_x[:, 0]
    else:
        history = seq_x[:, 0].numpy()

    # 真实未来：seq_y 的最后 PRED_LEN 步
    if isinstance(seq_y, np.ndarray):
        true_future = seq_y[-PRED_LEN:, 0]
    else:
        true_future = seq_y[-PRED_LEN:, 0].numpy()

    return seq_x, seq_y, seq_x_mark, seq_y_mark, history, true_future


def run_informer_single_sample(
    model: torch.nn.Module,
    dataset: Dataset_Custom,
    seq_x,
    seq_y,
    seq_x_mark,
    seq_y_mark,
    device: torch.device,
):
    """
    基于 Exp_Informer._process_one_batch 的逻辑，对单个样本做前向推理。
    """
    # 转为 batch_size=1
    batch_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).to(device)
    batch_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(0)
    batch_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32).unsqueeze(0).to(device)
    batch_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32).unsqueeze(0).to(device)

    # 构造 decoder 输入： [B, label_len + pred_len, C]
    # 先用真实的 label_len 作为起始，再接 pred_len 个全 0
    dec_inp = torch.zeros(
        [batch_y.shape[0], PRED_LEN, batch_y.shape[-1]], dtype=torch.float32
    )
    dec_inp = torch.cat([batch_y[:, :LABEL_LEN, :], dec_inp], dim=1).to(device)

    with torch.no_grad():
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    # 只取预测部分 [B, pred_len, C]
    # 与 Exp_Informer._process_one_batch 中 f_dim 的逻辑保持一致（S/M/MS 在这里都取第 0 维）
    pred = outputs[:, -PRED_LEN:, :]

    # 转回 CPU numpy
    pred = pred.detach().cpu().numpy()[0]  # [pred_len, C]
    return pred[:, 0]  # 单变量，取第 0 维


def inverse_transform_series(dataset: Dataset_Custom, series: np.ndarray):
    """
    使用 Dataset_Custom 内部的 StandardScaler 对一维序列做反归一化。
    """
    scaler = dataset.scaler
    # StandardScaler.inverse_transform 期望 2D，因此在序列上扩一维
    data_2d = series.reshape(-1, 1)
    inv_2d = scaler.inverse_transform(data_2d)
    return inv_2d.reshape(-1)


def visualize_sample(
    history_origin: np.ndarray,
    pred_origin: np.ndarray,
    true_origin: np.ndarray,
    sample_idx: int,
    vis_pred_len: int,
    fig_w: float,
    fig_h: float,
    ylabel: str,
):
    assert vis_pred_len <= PRED_LEN, "vis_pred_len 不能超过 24"

    # 只截取前 vis_pred_len 步的预测/真实
    pred_vis = pred_origin[:vis_pred_len]
    true_vis = true_origin[:vis_pred_len]

    total_len = SEQ_LEN + vis_pred_len
    time_axis = np.arange(total_len)

    # Prediction 曲线：前 168 点是真实历史，后 vis_pred_len 点是模型预测
    full_pred = np.concatenate([history_origin, pred_vis])

    # Ground Truth 曲线：前 168 点用 NaN 占位，后 vis_pred_len 点是真实未来
    gt_prefix = np.full(SEQ_LEN, np.nan, dtype=float)
    full_true = np.concatenate([gt_prefix, true_vis])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 背景白色，去掉网格（下面手动控制）
    ax.set_facecolor("white")

    # 预测线
    ax.plot(
        time_axis,
        full_pred,
        color="#f28e2b",
        linewidth=2.3,
        label="Prediction (history + forecast)",
    )

    # 真实线（预测区）
    ax.plot(
        time_axis,
        full_true,
        color="#4e79a7",
        linewidth=2.3,
        label="Ground Truth (forecast)",
    )

    ax.set_xlabel(
        f"Time steps ({SEQ_LEN} history + {vis_pred_len} forecast)",
        fontsize=12,
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"Informer Forecast Visualization\nsample_idx={sample_idx}, "
        f"vis_pred_len={vis_pred_len}",
        fontsize=14,
    )

    # 去掉默认网格
    ax.grid(False)

    # 图例：左上角，白底方框
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        fontsize=10,
    )
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()

    # 保存
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visualize_168_24")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_idx}.png")
    plt.savefig(out_path, dpi=300)
    print(f"可视化结果已保存到：{out_path}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Informer 单样本预测可视化脚本 (seq_len=168, pred_len=24)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(
            "checkpoints",
            "informer_PJME_ftS_sl168_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0",
            "checkpoint.pth",
        ),
        help="Informer 模型 checkpoint 路径 (.pth)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="数据根目录（与训练时 root_path 一致）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="PJME_hourly.csv",
        help="数据文件名（与训练时 data_path 一致）",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="S",
        help="特征模式: S / M / MS（需与训练时一致）",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="PJME_MW",
        help="目标列名（与训练时一致）",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="要可视化的测试样本索引 (0-based)",
    )
    parser.add_argument(
        "--vis_pred_len",
        type=int,
        default=24,
        help="在图中展示的预测步数，不能超过 24",
    )
    parser.add_argument(
        "--fig_w",
        type=float,
        default=12.0,
        help="图宽 (inch)，默认 12",
    )
    parser.add_argument(
        "--fig_h",
        type=float,
        default=9.0,
        help="图高 (inch)，默认 9",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Load (MW)",
        help="Y 轴标签，默认 'Load (MW)'",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="若指定则在可用时使用 GPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    )
    print(f"使用设备：{device}")

    if args.vis_pred_len > PRED_LEN:
        raise ValueError(f"vis_pred_len 不能超过 {PRED_LEN}，当前为 {args.vis_pred_len}")

    # 1. 构建并加载模型
    model = build_model(device)
    model = load_checkpoint(model, args.ckpt, device)

    # 2. 构建测试集（与训练切分/归一化保持一致）
    test_dataset = build_test_dataset(
        data_root=args.data_root,
        data_path=args.data_path,
        target=args.target,
        features=args.features,
    )

    # 3. 选取单个样本
    (
        seq_x,
        seq_y,
        seq_x_mark,
        seq_y_mark,
        history_scaled,
        true_future_scaled,
    ) = select_sample(test_dataset, args.sample_idx)

    # 4. 模型前向预测（在标准化尺度上）
    pred_scaled = run_informer_single_sample(
        model,
        test_dataset,
        seq_x,
        seq_y,
        seq_x_mark,
        seq_y_mark,
        device,
    )

    # 5. 反归一化到原始物理量
    history_origin = inverse_transform_series(test_dataset, history_scaled)
    pred_origin = inverse_transform_series(test_dataset, pred_scaled)
    true_origin = inverse_transform_series(test_dataset, true_future_scaled)

    # 6. 可视化
    visualize_sample(
        history_origin=history_origin,
        pred_origin=pred_origin,
        true_origin=true_origin,
        sample_idx=args.sample_idx,
        vis_pred_len=args.vis_pred_len,
        fig_w=args.fig_w,
        fig_h=args.fig_h,
        ylabel=args.ylabel,
    )


if __name__ == "__main__":
    main()

