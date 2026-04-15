import argparse
import json
import os

import numpy as np
import torch

from exp.exp_informer import Exp_Informer


def calc_metrics(pred, true, eps=1e-8):
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((pred - true) / np.maximum(np.abs(true), eps))))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Informer checkpoint with MAE/MSE/RMSE/MAPE on scaled and original scales."
    )

    parser.add_argument("--setting", type=str, required=True, help="experiment setting folder name")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint.pth", help="checkpoint file name")

    parser.add_argument("--model", type=str, required=True, default="informer")
    parser.add_argument("--data", type=str, required=True, default="PJME")
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--data_path", type=str, default="PJME_hourly.csv")
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="PJME_MW")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--s_layers", type=str, default="3,2,1")
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attn", type=str, default="prob")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true", default=False)
    parser.add_argument("--mix", action="store_false", default=True)
    parser.add_argument("--cols", type=str, nargs="+")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--save_dir", type=str, default="./results/")
    parser.add_argument("--eps", type=float, default=1e-8, help="small constant for stable MAPE")
    return parser


def apply_data_defaults(args):
    data_parser = {
        "ETTh1": {"data": "ETTh1.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
        "ETTh2": {"data": "ETTh2.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
        "ETTm1": {"data": "ETTm1.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
        "ETTm2": {"data": "ETTm2.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
        "WTH": {"data": "WTH.csv", "T": "WetBulbCelsius", "M": [12, 12, 12], "S": [1, 1, 1], "MS": [12, 12, 1]},
        "ECL": {"data": "ECL.csv", "T": "MT_320", "M": [321, 321, 321], "S": [1, 1, 1], "MS": [321, 321, 1]},
        "Solar": {"data": "solar_AL.csv", "T": "POWER_136", "M": [137, 137, 137], "S": [1, 1, 1], "MS": [137, 137, 1]},
        "PJME": {"root": "./data/", "data": "PJME_hourly.csv", "T": "PJME_MW", "M": [1, 1, 1], "S": [1, 1, 1], "MS": [1, 1, 1]},
    }
    if args.data in data_parser:
        info = data_parser[args.data]
        if "root" in info:
            args.root_path = info["root"]
        args.data_path = info["data"]
        args.target = info["T"]
        args.enc_in, args.dec_in, args.c_out = info[args.features]
    return args


def infer_scaled_predictions(exp, test_data, test_loader, args):
    preds = []
    trues = []

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            if args.padding == 0:
                dec_inp = torch.zeros(
                    [batch_y.shape[0], args.pred_len, batch_y.shape[-1]], device=exp.device
                ).float()
            else:
                dec_inp = torch.ones(
                    [batch_y.shape[0], args.pred_len, batch_y.shape[-1]], device=exp.device
                ).float()
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float()

            if args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == "MS" else 0
            true = batch_y[:, -args.pred_len :, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

    preds = np.array(preds).reshape(-1, args.pred_len, preds[0].shape[-1])
    trues = np.array(trues).reshape(-1, args.pred_len, trues[0].shape[-1])

    preds_raw = test_data.inverse_transform(preds)
    trues_raw = test_data.inverse_transform(trues)
    return preds, trues, preds_raw, trues_raw


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(i) for i in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    args.s_layers = [int(s) for s in args.s_layers.replace(" ", "").split(",")]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    args.inverse = False

    args = apply_data_defaults(args)

    exp = Exp_Informer(args)
    test_data, test_loader = exp._get_data(flag="test")

    ckpt_path = os.path.join(args.checkpoints, args.setting, args.checkpoint_file)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))

    pred_scaled, true_scaled, pred_raw, true_raw = infer_scaled_predictions(exp, test_data, test_loader, args)

    metrics_scaled = calc_metrics(pred_scaled, true_scaled, eps=args.eps)
    metrics_raw = calc_metrics(pred_raw, true_raw, eps=args.eps)

    result = {
        "setting": args.setting,
        "scaled": metrics_scaled,
        "original": metrics_raw,
    }

    out_dir = os.path.join(args.save_dir, args.setting)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics_dual_scale.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("===== Scaled Metrics =====")
    for k in ["MAE", "MSE", "RMSE", "MAPE"]:
        print(f"{k}: {metrics_scaled[k]:.10f}")
    print("===== Original Metrics =====")
    for k in ["MAE", "MSE", "RMSE", "MAPE"]:
        print(f"{k}: {metrics_raw[k]:.10f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
