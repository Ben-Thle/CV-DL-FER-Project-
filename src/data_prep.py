import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path

class ImageFolderDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir) 
        self.transform = transform
        self.images = []
        self.labels = [] 
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir() and d.name != 'neutral'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)} 


        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg']: 
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name]) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx] 

        image = Image.open(img_path).convert('L') 

        if self.transform:
            image = self.transform(image) 

        return image, label

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
base = project_root / "data" / "processed" / "split_data"

train_dataset = ImageFolderDataset(
    base / "train",
    transform=train_transform
)

val_dataset = ImageFolderDataset(
    base / "val",
    transform=test_transform
)

test_dataset = ImageFolderDataset(
    base / "test",
    transform=test_transform
)


train_dataset = ImageFolderDataset(
    '/FER2013_dataset/train',
    transform=train_transform
)

test_dataset = ImageFolderDataset(
    '/FER2013_dataset/test',
    transform=test_transform
)


