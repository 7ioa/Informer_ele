import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from informer_visualize_forecast import (
    build_model,
    load_checkpoint,
    build_test_dataset,
    select_sample,
    run_informer_single_sample,
    inverse_transform_series,
    SEQ_LEN,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Informer 序列一阶预测可视化 (按时间顺序取样)")
    parser.add_argument("--ckpt", type=str, default=os.path.join(
        "checkpoints",
        "informer_PJME_ftS_sl168_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0",
        "checkpoint.pth",
    ))
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--data_path", type=str, default="PJME_hourly.csv")
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="PJME_MW")
    parser.add_argument("--sample_idx", type=int, default=0, help="起始样本索引 (0-based)")
    parser.add_argument("--count", type=int, default=200, help="要连续取多少个样本进行比较，默认200")
    parser.add_argument("--fig_w", type=float, default=12.0)
    parser.add_argument("--fig_h", type=float, default=9.0)
    parser.add_argument("--ylabel", type=str, default="Load (MW)")
    parser.add_argument("--use_gpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"使用设备：{device}")

    # 1. 构建并加载模型
    model = build_model(device)
    model = load_checkpoint(model, args.ckpt, device)

    # 2. 构建测试集
    dataset = build_test_dataset(
        data_root=args.data_root,
        data_path=args.data_path,
        target=args.target,
        features=args.features,
    )

    n_samples = len(dataset)
    start = args.sample_idx
    if start < 0 or start >= n_samples:
        raise IndexError(f"sample_idx 超出范围：0 ~ {n_samples-1}")

    end = min(start + args.count, n_samples)
    actual_count = end - start
    if actual_count <= 0:
        raise ValueError("没有可用样本，请检查 sample_idx 和 count")

    preds_scaled = []
    trues_scaled = []

    # 3. 逐样本前向，仅取每次预测的第一步 (pred[0])
    for idx in range(start, end):
        seq_x, seq_y, seq_x_mark, seq_y_mark, history_scaled, true_future_scaled = select_sample(dataset, idx)
        pred_scaled = run_informer_single_sample(model, dataset, seq_x, seq_y, seq_x_mark, seq_y_mark, device)
        preds_scaled.append(pred_scaled[0])
        trues_scaled.append(true_future_scaled[0])

    preds_scaled = np.array(preds_scaled)
    trues_scaled = np.array(trues_scaled)

    # 4. 反归一化
    preds = inverse_transform_series(dataset, preds_scaled)
    trues = inverse_transform_series(dataset, trues_scaled)

    # 5. 绘图 — 按时间顺序比较预测(一步)与真实(对应第一步)
    time_axis = np.arange(start, start + actual_count)

    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))
    ax.set_facecolor("white")

    ax.plot(time_axis, preds, color="#f28e2b", linewidth=2.3, label="Prediction (1-step)")
    ax.plot(time_axis, trues, color="#4e79a7", linewidth=2.3, label="Ground Truth (1-step)")

    ax.set_xlabel(f"Samples (start={start}, count={actual_count})")
    ax.set_ylabel(args.ylabel)
    ax.set_title(f"Informer result (start={start}, count={actual_count})")

    legend = ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#333333", fontsize=10)
    legend.get_frame().set_alpha(0.9)
    ax.grid(False)

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visualize2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sequence_{start}_{actual_count}.png")
    plt.savefig(out_path, dpi=300)
    print(f"可视化结果已保存到：{out_path}")

    plt.show()


if __name__ == "__main__":
    main()
