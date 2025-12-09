import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np

class ExposureDataset(Dataset):
    def __init__(self, input_folder, gt_folder, size=512, transform=None):
        self.input_folder = input_folder
        self.gt_folder = gt_folder
        self.size = size
        self.transform = transform
        
        self.files = sorted([
            f for f in os.listdir(input_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"✓ Loaded {len(self.files)} images")
    
    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (self.size, self.size))
        return torch.tensor(img).permute(2, 0, 1).float()
    
    def _get_gt_path(self, input_name):
        stem = "_".join(input_name.split("_")[:-1]).split(".")[0]
        for ext in ['.jpg', '.jpeg', '.JPG', '.png']:
            gt_path = os.path.join(self.gt_folder, stem + ext)
            if os.path.exists(gt_path):
                return gt_path
        raise FileNotFoundError(f"GT not found for {input_name}")
    
    def __getitem__(self, idx):
        input_name = self.files[idx]
        input_path = os.path.join(self.input_folder, input_name)
        inp = self._load_image(input_path)
        gt_path = self._get_gt_path(input_name)
        gt = self._load_image(gt_path)
        if self.transform:
            inp = self.transform(inp)
            gt = self.transform(gt)
        return inp, gt
    
    def __len__(self):
        return len(self.files)

def get_data_loaders(train_input, train_gt, test_input, test_gt,
                    batch_size=32, num_workers=8, image_size=512):
    train_set = ExposureDataset(train_input, train_gt, size=image_size)
    test_set = ExposureDataset(test_input, test_gt, size=image_size)
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True
    )
    
    print(f"✓ Train: {len(train_set)} images, {len(train_loader)} batches")
    print(f"✓ Test: {len(test_set)} images, {len(test_loader)} batches")
    
    return train_loader, test_loader
