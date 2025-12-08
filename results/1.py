import os
import numpy as np
import matplotlib.pyplot as plt

# 推荐用正确路径，例：results/re/real_prediction.npy
path = os.path.join('results', 're', 'real_prediction.npy')  # 注意文件名和目录
if not os.path.exists(path):
    raise FileNotFoundError(f'File not found: {path}')

data = np.load(path)  # 若是用 pickle 保存才需要 allow_pickle=True
print('loaded:', path)
print('dtype:', data.dtype, 'shape:', data.shape)

# 根据维度选择要画的序列
if data.ndim == 1:
    series = data
elif data.ndim == 2:
    # 每列当一条曲线，或展平成一条序列： series = data.ravel()
    # 这里示例画第一列：
    series = data[:, 0]
elif data.ndim == 3:
    # 常见 shape: [num_samples, pred_len, num_features]
    # 选择：第一 sample 的第一特征时间序列
    series = data[0, :, 0]
    # 或者把所有样本和时间展平成一维： series = data.reshape(-1, data.shape[-1])[:, 0]
else:
    raise ValueError('Unsupported array ndim: {}'.format(data.ndim))

plt.figure(figsize=(8,4))
plt.plot(series)
plt.title(f'Plot of {os.path.basename(path)} (shape {data.shape})')
plt.xlabel('time index')
plt.ylabel('value')
plt.grid(True)
plt.show()