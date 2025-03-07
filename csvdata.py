import os
import torch
import numpy as np
from torch.utils.data import Dataset
from masking_generator import TubeMaskingGenerator

class CSVMAE(Dataset):
    def __init__(self, root_dir, num_frames=16, mask_ratio=0.75, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.mask_generator = TubeMaskingGenerator(input_size=(num_frames, 4, 4), mask_ratio=mask_ratio)

        # 遞迴地獲取所有 CSV 檔案
        self.file_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in sorted(files):  # 確保按名稱排序，維持時間順序
                if file.endswith('.csv'):
                    self.file_paths.append(os.path.join(subdir, file))

        if len(self.file_paths) < num_frames:
            raise ValueError(f"CSV 檔案數量不足 {num_frames}，請檢查 `{root_dir}` 資料夾！")

        if len(self.file_paths) < num_frames:
            raise ValueError(f"資料夾內的 CSV 檔案數量不足 {num_frames}，請檢查資料！")

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
        mask = torch.tensor(mask, dtype=torch.bool).reshape(self.num_frames, 4, 4)

        return (frames, mask)
