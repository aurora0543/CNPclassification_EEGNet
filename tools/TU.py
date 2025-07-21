import mne
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

root_dir = Path("/Volumes/Public/data_keri")
mne.set_log_level('ERROR')

set_files = list(root_dir.rglob("*.set"))

set_files = [f for f in set_files if "Unused AB" not in f.parts]
set_files = [f for f in set_files if "RS-AB" not in f.parts]
set_files = [f for f in set_files if not f.name.endswith("baseline.set")]
print(f"find {len(set_files)} Files")

labels = set()
for f in set_files:
    try:
        if "AB" in f.parts:
            labels.add("AB")
        elif  "PdP" in f.parts:
            labels.add("PdP")
        elif "PnP" in f.parts:
            labels.add("PnP")
        elif "PwP" in f.parts:
            labels.add("PwP")
        else:
            print(f"Unknown label in {f.parent}")
    except Exception as e:
        print(f"Error processing {f}: {e}")

# varify labels

for l in labels:
    if l not in ["AB", "PdP", "PnP", "PwP"]:
        print(f"Unknown label: {l}")

label_mapping = {lbl: idx for idx, lbl in enumerate(sorted(labels))}
print("Label mapping:", label_mapping)

X_all = []
y_all = []


for file_path in tqdm(set_files, desc="Processing files"):
    try:
        # 尝试先用 read_epochs_eeglab
        try:
            data_obj = mne.io.read_epochs_eeglab(str(file_path))
        except Exception:
            # 如果不是 epochs 文件，就用 raw 读
            data_obj = mne.io.read_raw_eeglab(str(file_path), preload=True)

        # ----------- 提取 label -----------
        if "AB" in file_path.parts:
            label = "AB"
        elif "PdP" in file_path.parts:
            label = "PdP"
        elif "PnP" in file_path.parts:
            label = "PnP"
        elif "PwP" in file_path.parts:
            label = "PwP"
        else:
            print(f"Unknown label in {file_path.parent}")
            continue

        label_idx = label_mapping[label]

        # ----------- 如果是 Raw 类型 -----------
        if isinstance(data_obj, mne.io.BaseRaw):
            raw = data_obj

            # Picks
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            raw.pick(picks)

            # Average reference
            raw.set_eeg_reference('average')

            # Filter
            raw.filter(1., 40., fir_design='firwin')

            # Epoch
            epoch_length = 1.0
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=epoch_length,
                overlap=0.0,
                preload=True,
                reject_by_annotation=True
            )
        
        # ----------- 如果是 Epochs 类型 -----------
        elif isinstance(data_obj, mne.BaseEpochs):
            epochs = data_obj

            # Picks
            picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
            epochs.pick(picks)

            # Average reference
            epochs.set_eeg_reference('average')

            # Filter
            epochs.filter(1., 40., fir_design='firwin')

            # 如果 epochs 太长，也可以再切成 1s
            epoch_length = 1.0
            epochs = mne.make_fixed_length_epochs(
                epochs,
                duration=epoch_length,
                overlap=0.0,
                preload=True,
                reject_by_annotation=True
            )

        else:
            print(f"Unknown data type in {file_path}")
            continue

        # ----------- 提取数据 -----------
        data = epochs.get_data()

        # z-score per epoch
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        data = (data - mean) / (std + 1e-8)

        # save
        X_all.append(data)
        y_all.append(np.full((data.shape[0],), label_idx))

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# 拼接数据
if len(X_all) > 0:
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    print(f"Final data shape: X = {X_all.shape}, y = {y_all.shape}")
else:
    print("No data processed.")