# dataloader.py

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import gc
from pathlib import Path

class EEGDataModule:
    """
    一个用于处理、标准化和加载EEG数据的模块。

    该类封装了从 .npz 文件加载数据、计算标准化统计数据、
    创建 PyTorch TensorDataset 和提供 DataLoader 的所有逻辑。
    """
    def __init__(self, data_dir: str, batch_size: int = 64):
        """
        初始化数据模块。

        Args:
            data_dir (str): 存放 eeg_{train|val|test}.npz 文件的目录。
            batch_size (int): DataLoader 使用的批次大小。
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        self.mean = None
        self.std = None

    def setup(self):
        """
        执行完整的数据加载和预处理流程。
        这个方法应该在创建 DataLoader 之前被调用一次。
        """
        print("--- Setting up EEGDataModule ---")
        # 1. 加载数据
        X_train, y_train = self._load_data("train")
        X_val, y_val = self._load_data("val")
        X_test, y_test = self._load_data("test")

        # 2. 转换为张量
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).long()
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).long()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()
        
        # 释放 NumPy 内存
        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()
        print("NumPy memory released.")

        # 3. 计算标准化统计数据 (仅从训练集)
        self._calculate_stats(X_train_tensor)

        # 4. 原地标准化所有数据
        self._standardize_inplace(X_train_tensor)
        self._standardize_inplace(X_val_tensor)
        self._standardize_inplace(X_test_tensor)
        print("All datasets standardized.")
        
        # 5. 创建 TensorDataset
        self.train_set = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_set = TensorDataset(X_val_tensor, y_val_tensor)
        self.test_set = TensorDataset(X_test_tensor, y_test_tensor)
        print("TensorDatasets created.")
        
        # 再次释放不再需要的完整张量
        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
        gc.collect()
        print("PyTorch tensor memory released.")
        print("--- Setup complete ---")


    def _load_data(self, split: str):
        """从 .npz 文件加载数据。"""
        print(f"Loading {split} data...")
        file_path = self.data_dir / f"eeg_{split}.npz"
        data = np.load(file_path)
        X = data["X"]
        y = data["y"]
        print(f"  - Shape: X={X.shape}, y={y.shape}")
        return X, y

    def _calculate_stats(self, X_train_tensor):
        """计算均值和标准差。"""
        # N=dim0, Freq=dim1, Chan=dim2, Time=dim3
        # 在 dim 0, 1, 3 上计算，保留 dim 2 (通道维度)
        self.mean = X_train_tensor.mean(dim=(0, 1, 3))
        self.std = X_train_tensor.std(dim=(0, 1, 3)) + 1e-6 # 加上 epsilon 防止除以零
        print(f"Calculated mean (shape: {self.mean.shape}) and std (shape: {self.std.shape}) from training data.")

    def _standardize_inplace(self, X_tensor):
        """对张量进行原地标准化。"""
        # mean 和 std 的形状为 (Chan,)
        # 调整形状以利用广播机制: (1, 1, Chan, 1)
        mean_reshaped = self.mean.view(1, 1, -1, 1)
        std_reshaped = self.std.view(1, 1, -1, 1)
        
        X_tensor.sub_(mean_reshaped).div_(std_reshaped)

    def train_dataloader(self):
        """获取训练数据的 DataLoader。"""
        if not self.train_set:
            raise RuntimeError("Please call setup() before creating dataloaders.")
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """获取验证数据的 DataLoader。"""
        if not self.val_set:
            raise RuntimeError("Please call setup() before creating dataloaders.")
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """获取测试数据的 DataLoader。"""
        if not self.test_set:
            raise RuntimeError("Please call setup() before creating dataloaders.")
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)