import os
import torch
import numpy as np
from torch.utils.data import Dataset
from masking_generator import TubeMaskingGenerator
import re

class CSVMAE(Dataset):
    def __init__(self, root_dir, num_frames=16, mask_ratio=0.75, window_size=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.window_size = window_size
        self.mask_generator = TubeMaskingGenerator(window_size, mask_ratio=mask_ratio)

        # 遞迴地獲取所有 CSV 檔案
        self.file_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv'):
                    self.file_paths.append(os.path.join(subdir, file))

        # 自訂排序函數: 先按類別名稱排序，再按數字時間戳排序
        def extract_sort_key(path):
            filename = os.path.basename(path)
            match = re.match(r"(.+?)_(\d+)\.csv", filename)  # 擷取類別名稱 & 數字
            if match:
                category = match.group(1)  # 取得類別名稱 (如 1119-1in3out-mv-mc-mid)
                time_value = int(match.group(2))  # 取得數字時間戳
                return (category, time_value)
            return (filename, 0)  # 若無法匹配則放最後

        # 進行排序
        self.file_paths.sort(key=extract_sort_key)

        # 檢查 CSV 數量是否足夠
        if len(self.file_paths) < num_frames:
            raise ValueError(f"CSV 檔案數量不足 {num_frames}，請檢查 `{root_dir}` 資料夾！")
        
        # 顯示前 10 個檔案確認順序
        print("=加載的 CSV 檔案順序:")
        for i, path in enumerate(self.file_paths[:10]):
            print(f"{i+1}. {os.path.basename(path)}")

    def __len__(self):
        return len(self.file_paths) - self.num_frames + 1

    def __getitem__(self, idx):
        frames = []
        for i in range(self.num_frames):
            csv_path = self.file_paths[idx + i]
            data = np.genfromtxt(csv_path, delimiter=',')
            frames.append(data)

        frames = np.stack(frames)  # (T, H, W)

        # 增加 Channel 維度 -> (C=1, T, H, W)
        frames = np.expand_dims(frames, axis=0)
        frames = torch.tensor(frames, dtype=torch.float32)

        # 生成 mask
        mask = self.mask_generator()

        return (frames, mask)