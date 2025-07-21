from pathlib import Path
import mne
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

root_dir = Path("/Volumes/Public/data_keri")

# find motor imagery and resting state files and exclude unused files
set_files_all = list(root_dir.rglob("*.set"))
set_files_all = [f for f in set_files_all if "Unused AB" not in f.parts]
set_files_RS = list(root_dir.rglob("*_EC.set")) + list(root_dir.rglob("*_EO.set")) + list(root_dir.rglob("*_baseline.set"))
set_files_RS = [f for f in set_files_RS if "Unused AB" not in f.parts]
set_files_MI = [f for f in set_files_all if f not in set_files_RS and f.suffix == ".set"]

print(f"Find {len(set_files_all)} Files in Total")
print(f"Find {len(set_files_RS)} Resting State Files")
print(f"Find {len(set_files_MI)} Motor Imagery Files")

mne.set_log_level('ERROR')  # 或 warning
labelled_patients = {'PwP' : 0, 'PnP' : 1, 'PdP' : 2, 'AB' : 3}

def load_all_epochs(file_list, set_table):
    """
    Load all epochs from a list of files and assign labels based on the provided set_table.
    Args:
        file_list (list): List of file paths to load epochs from.
        set_table (dict): Dictionary mapping labels to their corresponding values.
    Returns:
        epochs (list): List of loaded epochs.
        labels (list): List of labels corresponding to the loaded epochs.
    """
    epochs = []
    labels = []
    for f in tqdm(file_list, desc="Loading all epoch files"):
        try:
            ep = mne.read_epochs_eeglab(str(f))
            
            label = [key for key in set_table.keys() if key in f.parts]
            if label:
                label_value = set_table[label[0]]
                epochs.append(ep)
                labels.append(label_value)
            else:
                print(f"No label found for file: {f}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    return epochs, labels


def sliding_window_epochs(ep, window_s=2.0, step_s=2.0):
    """
    将 mne.Epochs 对象按长度 window_s 和步长 step_s 划分为多个短片段。
    输出 shape = (n_windows, n_channels, n_times_window)
    """
    fs = ep.info['sfreq']
    win_len = int(window_s * fs)
    step = int(step_s * fs)
    data = ep.get_data()  # shape = (n_trials, n_channels, n_times)
    
    windows = []
    for trial in data:
        max_start = trial.shape[1] - win_len
        for st in range(0, max_start + 1, step):
            windows.append(trial[:, st:st + win_len])
    return np.array(windows)



def eeg_epochs_preprocess(eppchs_data):
    """
    Preprocess epochs data by applying filtering and cropping.
    Args:
        epochs_data (mne.Epochs): Epochs data to preprocess.
    Returns:
        epochs_data (mne.Epochs): Preprocessed epochs data.
    """
    eppchs_data = eppchs_data.copy().crop(tmin=0, tmax=eppchs_data.times[-1])  # remove baseline
    eppchs_data.filter(l_freq=0.1, h_freq=None, fir_design='firwin', skip_by_annotation='edge')  # high pass filter 0.1 HZ
    eppchs_data.notch_filter(freqs=50, fir_design='firwin', skip_by_annotation='edge')  # remove 50Hz noise
    eppchs_data.set_eeg_reference(projection=True)  # 设置参考电极

    return eppchs_data

epochs, labels = load_all_epochs(set_files_MI, labelled_patients)
x, y = eeg_epochs_preprocess(epochs), labels
print(f"Loaded {len(x)} windows with shape {x.shape} and labels {np.unique(y)}")