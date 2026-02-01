import sys
from pathlib import Path

print("RUN_GRADCAM: file executed")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import torch
from PIL import Image
from src.data_prep import ImageFolderDataset
from torchvision import transforms

from src.models.model import build_model
from src.explainability.gradcam import GradCAM, overlay_red, load_checkpoint_flexible


def main():
    print("RUN_GRADCAM: main started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = ImageFolder(ROOT / "data" / "processed"/ "split_data", transform=tfm)
    print("RUN_GRADCAM: classes =", dataset.classes)

    model = build_model(
        name="resnet18",
        num_classes=len(dataset.classes),
        input_channels=1,
        small_input=True,
    ).to(device)

    ckpt = ROOT / "best_model.pt"
    if not ckpt.exists():
        
        alt = ROOT / "experiments_finetuned" / "best_model.pt"
        if alt.exists():
            ckpt = alt
    print("RUN_GRADCAM: ckpt =", ckpt)

    load_checkpoint_flexible(model, str(ckpt), device)
    model.eval()

    img, y = dataset[0]
    x = img.unsqueeze(0).to(device)

    target_layer = model.layer4[-1].conv2
    gc = GradCAM(model, target_layer)
    res = gc(x)
    gc.close()

    pred_name = dataset.classes[res.class_idx]
    true_name = dataset.classes[int(y)]
    print(f"RUN_GRADCAM: true={true_name} pred={pred_name} logit={res.score:.4f}")
    img_cpu=img.detach().cpu()
    img_denormalized=(img_cpu*0.5+0.5).clamp(0,1)

    pil_gray = Image.fromarray((img_denormalized[0].numpy() * 255).astype("uint8"))
    pil_gray = pil_gray.resize((x.shape[-1], x.shape[-2]))

    overlay = overlay_red(pil_gray.convert("RGB"), res.cam, alpha=0.45)
    out_path = ROOT / f"gradcam_idx0_true_{true_name}_pred_{pred_name}.png"
    overlay.save(out_path)
    print("RUN_GRADCAM: saved", out_path)

#für ein problem das ich hatte beim Runnen
if __name__ == "__main__":
    main()
