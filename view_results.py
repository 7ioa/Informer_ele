"""
Informer 训练结果查看工具
用于查看和可视化 .npy 格式的训练结果文件
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_npy_file(file_path):
    """加载 npy 文件并返回数据"""
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"加载文件失败：{e}")
        return None

def view_metrics(metrics_path):
    """查看评估指标"""
    print("\n" + "="*60)
    print("📊 评估指标")
    print("="*60)
    
    metrics = load_npy_file(metrics_path)
    if metrics is None:
        return
    
    metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
    print(f"\n文件路径：{metrics_path}")
    print(f"数据形状：{metrics.shape}")
    print(f"\n详细指标:")
    print("-" * 60)
    for name, value in zip(metric_names, metrics):
        print(f"{name:8s}: {value:.6f}")
    print("-" * 60)
    
    return metrics

def view_predictions(pred_path, true_path=None):
    """查看预测结果"""
    print("\n" + "="*60)
    print("📈 预测结果分析")
    print("="*60)
    
    preds = load_npy_file(pred_path)
    if preds is None:
        return
    
    print(f"\n文件路径：{pred_path}")
    print(f"数据形状：{preds.shape}")
    print(f"  - 样本数：{preds.shape[0]}")
    print(f"  - 预测长度：{preds.shape[1]}")
    if len(preds.shape) > 2:
        print(f"  - 变量数：{preds.shape[2]}")
    
    print(f"\n统计信息:")
    print(f"  最小值：{preds.min():.6f}")
    print(f"  最大值：{preds.max():.6f}")
    print(f"  平均值：{preds.mean():.6f}")
    print(f"  标准差：{preds.std():.6f}")
    
    # 如果提供了真实值，进行对比
    if true_path and os.path.exists(true_path):
        trues = load_npy_file(true_path)
        if trues is not None:
            print(f"\n真实值统计:")
            print(f"  最小值：{trues.min():.6f}")
            print(f"  最大值：{trues.max():.6f}")
            print(f"  平均值：{trues.mean():.6f}")
            print(f"  标准差：{trues.std():.6f}")
            
            # 计算误差
            error = np.abs(preds - trues)
            print(f"\n绝对误差统计:")
            print(f"  平均绝对误差：{error.mean():.6f}")
            print(f"  最大绝对误差：{error.max():.6f}")
    
    return preds

def plot_predictions(pred_path, true_path=None, sample_idx=0, variable_idx=0):
    """可视化预测结果"""
    print("\n" + "="*60)
    print("📉 生成可视化图表")
    print("="*60)
    
    preds = load_npy_file(pred_path)
    if preds is None:
        return
    
    # 确保是 3D 数据
    if len(preds.shape) == 2:
        preds = preds.reshape(-1, preds.shape[-2], 1)
    
    time_steps = preds.shape[1]
    
    # 创建时间轴
    time_axis = np.arange(time_steps)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制预测值
    if len(preds.shape) > 2:
        pred_line = preds[sample_idx, :, variable_idx]
    else:
        pred_line = preds[sample_idx, :]
    
    plt.plot(time_axis, pred_line, 'r-', label='Prediction', linewidth=2, alpha=0.7)
    
    # 如果有真实值，绘制真实值
    if true_path and os.path.exists(true_path):
        trues = load_npy_file(true_path)
        if trues is not None:
            if len(trues.shape) == 2:
                trues = trues.reshape(-1, trues.shape[-2], 1)
            
            if len(trues.shape) > 2:
                true_line = trues[sample_idx, :, variable_idx]
            else:
                true_line = trues[sample_idx, :]
            
            plt.plot(time_axis, true_line, 'b-', label='True', linewidth=2, alpha=0.7)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(f'Prediction vs True (Sample {sample_idx}, Variable {variable_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    save_path = pred_path.replace('.npy', '_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至：{save_path}")
    
    plt.show()

def compare_pred_true(pred_path, true_path):
    """详细对比预测值和真实值"""
    print("\n" + "="*60)
    print("🔍 预测值 vs 真实值 详细对比")
    print("="*60)
    
    preds = load_npy_file(pred_path)
    trues = load_npy_file(true_path)
    
    if preds is None or trues is None:
        return
    
    if preds.shape != trues.shape:
        print(f"⚠️  警告：预测值和真实值形状不匹配!")
        print(f"   预测值形状：{preds.shape}")
        print(f"   真实值形状：{trues.shape}")
        return
    
    print(f"\n数据形状：{preds.shape}")
    
    # 逐时间点分析
    if len(preds.shape) == 3:
        n_samples, n_steps, n_vars = preds.shape
        print(f"\n时间维度分析 (前 5 个时间步):")
        print("-" * 60)
        for t in range(min(5, n_steps)):
            pred_mean = preds[:, t, :].mean()
            true_mean = trues[:, t, :].mean()
            error = np.abs(pred_mean - true_mean)
            print(f"t={t:2d}: 预测均值={pred_mean:8.4f}, 真实均值={true_mean:8.4f}, 误差={error:8.4f}")
        print("-" * 60)
        
        # 逐变量分析
        if n_vars > 1:
            print(f"\n变量维度分析 (共{n_vars}个变量):")
            print("-" * 60)
            for v in range(n_vars):
                pred_mean = preds[:, :, v].mean()
                true_mean = trues[:, :, v].mean()
                mae = np.abs(preds[:, :, v] - trues[:, :, v]).mean()
                print(f"变量{v:2d}: 预测均值={pred_mean:8.4f}, 真实均值={true_mean:8.4f}, MAE={mae:8.4f}")
            print("-" * 60)

def main():
    """主函数"""
    print("="*60)
    print("🔬 Informer 训练结果查看工具")
    print("="*60)
    
    # 示例目录（可以修改为实际路径）
    results_dir = 'results\informer_PJME_ftS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_PJME_0'
    
    if not os.path.exists(results_dir):
        print(f"\n❌ 目录不存在：{results_dir}")
        print("请修改 results_dir 变量为正确的路径")
        return
    
    print(f"\n📂 结果目录：{results_dir}")
    
    # 查找所有 npy 文件
    npy_files = [f for f in os.listdir(results_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"\n❌ 未找到 .npy 文件")
        return
    
    print(f"\n找到 {len(npy_files)} 个 .npy 文件:")
    for i, file in enumerate(npy_files, 1):
        print(f"  {i}. {file}")
    
    # 自动识别文件类型
    metrics_file = None
    pred_file = None
    true_file = None
    real_pred_file = None
    
    for file in npy_files:
        if 'metrics' in file:
            metrics_file = os.path.join(results_dir, file)
        elif 'pred' in file and 'real' not in file:
            pred_file = os.path.join(results_dir, file)
        elif 'true' in file:
            true_file = os.path.join(results_dir, file)
        elif 'real' in file:
            real_pred_file = os.path.join(results_dir, file)
    
    # 查看各类文件
    if metrics_file:
        view_metrics(metrics_file)
    
    if pred_file:
        view_predictions(pred_file, true_file)
        
        if true_file:
            compare_pred_true(pred_file, true_file)
            
            # 询问是否生成可视化
            choice = input("\n是否生成可视化图表？(y/n): ")
            if choice.lower() == 'y':
                plot_predictions(pred_file, true_file)
    
    if real_pred_file:
        print("\n" + "="*60)
        print("🔮 未来预测结果")
        print("="*60)
        view_predictions(real_pred_file)

if __name__ == "__main__":
    main()
