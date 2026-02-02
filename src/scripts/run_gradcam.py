import sys
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class FERFolderCSVDataset(Dataset):
    def __init__(self, split_root, class_names, transform=None):
        self.transform = transform
        self.samples = []

        self.classes = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for label, cname in enumerate(self.classes):
            csv_path = split_root / cname / "data.csv"
            df = pd.read_csv(csv_path)

            if "pixels" in df.columns:
                for p in df["pixels"].astype(str):
                    arr = np.fromstring(p, sep=" ", dtype=np.uint8)
                    self.samples.append((arr, label))
            else:
                num_df = df.select_dtypes(include=[np.number])
                for _, row in num_df.iterrows():
                    arr = row.to_numpy(dtype=np.uint8)
                    self.samples.append((arr, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]

        n = int(np.sqrt(len(arr)))
        if n * n != len(arr):
            raise ValueError(f"Cannot reshape array of length {len(arr)} into square image")

        img = arr.reshape(n, n)
        pil = Image.fromarray(img, mode="L")  
        if self.transform:
            x = self.transform(pil)
        else:
            x = pil

        return x, label


print("RUN_GRADCAM: file executed")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import torch
from PIL import Image

from torchvision import transforms

from src.models.model import build_model
from src.explainability.gradcam import GradCAM, overlay_red, load_checkpoint_flexible


def main():
    print("RUN_GRADCAM: main started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
    ])
    class_names = ["angry","disgust","fear","happy","sad","surprise"]
    split_root = ROOT / "data" / "processed" / "split_data" / "train"
    dataset = FERFolderCSVDataset(split_root, class_names, transform=tfm)
    print("RUN_GRADCAM: classes =", class_names)


    model = build_model(
        name="resnet18",
        num_classes=len(dataset.classes),
        input_channels=1,
        small_input=True,
    ).to(device)

    ckpt = ROOT / "src" / "camDemo" / "checkpoint_epoch_61.pt"
    if not ckpt.exists():
        
        alt = ROOT / "experiments_finetuned" / "best_model.pt"
        if alt.exists():
            ckpt = alt
    print("RUN_GRADCAM: ckpt =", ckpt)

    load_checkpoint_flexible(model, str(ckpt), device)
    model.eval()

    

    target_layer = model.layer4[-1].conv2
    gc = GradCAM(model, target_layer)

    out_dir = ROOT / "gradcam_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(10, len(dataset))):
        img, y = dataset[idx]
        x = img.unsqueeze(0).to(device)

        res = gc(x)

        pred_name = dataset.classes[res.class_idx]
        true_name = dataset.classes[int(y)]
        print(f"[{idx}] true={true_name} pred={pred_name} logit={res.score:.4f}")

        img_cpu = img.detach().cpu()
        img_denormalized = (img_cpu * 0.5 + 0.5).clamp(0, 1)

        pil_gray = Image.fromarray(
            (img_denormalized[0].numpy() * 255).astype("uint8"),
            mode="L"
        )

        overlay = overlay_red(pil_gray.convert("RGB"), res.cam, alpha=0.45)

        out_path = out_dir / f"idx{idx:03d}_true_{true_name}_pred_{pred_name}.png"
        overlay.save(out_path)
        print("saved:", out_path)

    gc.close()
if __name__ == "__main__":
    main()
