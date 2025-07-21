from pathlib import Path
import mne
import numpy as np
from tqdm import tqdm

root_dir = Path("/Volumes/Public/data_keri")
mne.set_log_level('ERROR')  # 或 'ERROR'
# 匹配所有 *_EC.set 和 *_EO.set 文件
set_files = list(root_dir.rglob("*_EC.set")) + list(root_dir.rglob("*_EO.set"))

# 正确排除包含 "Unused" 的路径
set_files = [f for f in set_files if "Unused AB" not in f.parts]


print(f"Find {len(set_files)} Files")

# 先收集所有 label
labels_found = set()
for file_path in set_files:
    # 这里假设 label 在 parent.parent 层
    # 如 /Volumes/Public/data_keri/AB/AB01/AB01_EC.set
    label = file_path.parent.parent.parent.name
    labels_found.add(label)

# 建立 label → 数字 映射
label_mapping = {lbl: idx for idx, lbl in enumerate(sorted(labels_found))}
print("Label mapping:", label_mapping)

# 存储所有数据
X_all = []
y_all = []

for file_path in tqdm(set_files, desc="Processing files"):
    raw = mne.io.read_raw_eeglab(str(file_path), preload=True)
    
    # 提取 label
    label = file_path.parent.parent.parent.name
    label_idx = label_mapping[label]

    # picks
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    raw.pick(picks)
    
    # average reference
    raw.set_eeg_reference('average')

    # filter
    raw.filter(1., 40., fir_design='firwin')

    # epoch
    epoch_length = 1.0
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=epoch_length,
        overlap=0.0,
        preload=True,
        reject_by_annotation=True
    )
    data = epochs.get_data()   # (n_epochs, n_channels, n_times)

    # z-score per epoch
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    data = (data - mean) / std

    # save
    X_all.append(data)
    y_all.append(np.full((data.shape[0],), label_idx))

# 合并所有 subject
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print("Final shape:", X_all.shape)
print("Labels shape:", y_all.shape)


np.savez("preprocessed_data.npz", X=X_all, y=y_all)
print("Dataset Svaed as preprocessed_data.npz")