import numpy as np
import matplotlib.pyplot as plt

# 参数设置
t = np.linspace(0, 3, 1500)    # 横轴时间 0-3 秒
channels = 4
offsets = np.arange(channels) * 0.5

# Epoch 设置
epoch_length_sec = 1.0
num_epochs = int(t[-1] / epoch_length_sec)
epoch_boundaries = np.arange(0, t[-1]+epoch_length_sec, epoch_length_sec)

# 创建画布
fig, ax = plt.subplots(figsize=(6, 2))

# 绘制多通道 EEG
for i in range(channels):
    freq = 5 + i
    y = np.sin(2 * np.pi * freq * t) * 0.2 + offsets[i]
    ax.plot(t, y, lw=2)

# 绘制 epoch 分割线
for b in epoch_boundaries:
    ax.axvline(b, color='gray', linestyle='--', linewidth=1)

# 可选：加 epoch label
for i, b in enumerate(epoch_boundaries[:-1]):
    center = b + epoch_length_sec/2
    ax.text(center, offsets[-1] + 0.3, f"Epoch {i+1}",
            ha='center', va='center', fontsize=8, color='black')

# 去掉坐标轴
ax.axis('off')

# 设置坐标范围，避免多余白边
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-0.3, offsets[-1] + 0.5)

# 保存为 SVG - 透明背景
plt.savefig(
    "eeg_epochs.svg",
    bbox_inches='tight',    # 边框紧凑
    pad_inches=0.01,        # 去掉多余 padding
    transparent=True        # 背景透明
)
plt.close()