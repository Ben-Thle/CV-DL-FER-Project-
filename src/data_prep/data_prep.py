import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import base64
import io


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

        self.image_data = []  # Store base64 strings or image data
        self.labels = []

        # Load from CSV files with base64-encoded images
        for class_name in self.class_names:
            csv_path = self.root_dir / class_name / "data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    img_data = row['image_data']  # Base64-encoded image string
                    self.image_data.append(img_data)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_data = self.image_data[idx]
        label = self.labels[idx]

        # Decode base64 image using the provided code
        img_bytes = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(img_bytes)).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


# transforms (grayscale: mean/std have 1 value) --
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(degrees=15),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


#  dataset paths
project_root = Path(__file__).resolve().parents[1]
base = project_root / "data" / "processed" / "split_data"

train_dataset = ImageFolderDataset(base / "train", transform=train_transform)
val_dataset   = ImageFolderDataset(base / "val",   transform=test_transform)
test_dataset  = ImageFolderDataset(base / "test",  transform=test_transform)

