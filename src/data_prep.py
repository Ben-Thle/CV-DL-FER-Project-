import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ImageFolderDataset(Dataset):
  
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # class folders (ignore 'neutral' if it exists)
        self.class_names = sorted(
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and d.name != "neutral"
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        self.images = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

            for img_path in class_dir.glob("*.jpeg"):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # If you want grayscale training, keep 'L' (1 channel):
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


# transforms (grayscale: mean/std have 1 value) --
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


#  dataset paths
project_root = Path(__file__).resolve().parents[2]
base = project_root / "data" / "processed" / "split_data"

train_dataset = ImageFolderDataset(base / "train", transform=train_transform)
val_dataset   = ImageFolderDataset(base / "val",   transform=test_transform)
test_dataset  = ImageFolderDataset(base / "test",  transform=test_transform)
