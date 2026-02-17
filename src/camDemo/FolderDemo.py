import os
import glob
import csv
import torch
from torchvision import transforms
from src.models.model import build_model
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np


device = torch.device("cpu")

mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40,thresholds=[0.6, 0.7, 0.7], factor=0.709,post_process=True, device=device)


input_folder = r"D:\Uni\Semester III\Praktikum\Projekt\3\TestImages"
output_location = r"D:\Uni\Semester III\Praktikum\Projekt\3\results.csv"

checkpoint = torch.load("src/camDemo/checkpoint_epoch_61.pt", map_location='cpu')
num_classes = 6
state_dict = None
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('model')
else:
    state_dict = checkpoint

model = build_model("resnet18", num_classes=num_classes, input_channels=1, small_input=True)

if state_dict is not None:
    try:
        model.load_state_dict(state_dict)
        print("Model weights loaded from checkpoint.")
    except RuntimeError as e:
        print("Direct load failed:", e)
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k.replace('module.', '')] = v

        fc_keys = ['fc.weight', 'fc.bias']
        mismatch = False
        for key in fc_keys:
            if key in new_state and key in dict(model.named_parameters()):
                if new_state[key].shape != dict(model.named_parameters())[key].shape:
                    print(f"Shape mismatch for {key}: checkpoint {new_state[key].shape} vs model {dict(model.named_parameters())[key].shape}")
                    mismatch = True

        if mismatch:
            for key in fc_keys:
                if key in new_state:
                    print(f"Removing {key} from checkpoint before loading")
                    del new_state[key]
            model.load_state_dict(new_state, strict=False)
            print("Model weights loaded with final layer skipped.")
        else:
            model.load_state_dict(new_state)
            print("Model weights loaded after stripping 'module.' prefix.")

model.to(device)
model.eval()

if isinstance(checkpoint, dict) and checkpoint.get('class_names'):
    labels = checkpoint.get('class_names')
else:
    labels = ["angry", "disgust", "fear", "happy", "sad", "surprise"]


def determine_emotion_from_face_array(face_array):
    try:
        img = Image.fromarray(face_array).convert("L")
    except Exception:
        return None

    transform_local = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_tensor = transform_local(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    return probs


def process_image(image):
    boxes, scores = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return None, None

    idx = int(np.argmax(scores)) if scores is not None else 0
    x1, y1, x2, y2 = map(int, boxes[idx])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    face = image.crop((x1, y1, x2, y2)).resize((64, 64)).convert("L")
    face64 = np.array(face)

    return face64, (x1, y1, x2, y2)


def image_files_in_folder(folder):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files)
    return files


def run_folder_to_csv(input_folder, output_csv):
    files = image_files_in_folder(input_folder)
    if not files:
        print(f"No image files found in {input_folder}")
        return

    columns = ['filepath', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    columns_to_label = {
        'happiness': 'happy',
        'surprise': 'surprise',
        'sadness': 'sad',
        'anger': 'angry',
        'disgust': 'disgust',
        'fear': 'fear'
    }

    label_to_idx = {lab: idx for idx, lab in enumerate(labels)}

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(columns)

        for fp in files:
            try:
                img = Image.open(fp).convert('RGB')
            except Exception as e:
                print(f"Failed to open {fp}: {e}")
                continue

            face_array, box = process_image(img)
            if face_array is None:
                # if no face is detected
                row = [fp] + [0.0] * (len(columns) - 1)
                writer.writerow(row)
                print(f"No face detected: {fp}")
                continue

            probs = determine_emotion_from_face_array(face_array)
            if probs is None:
                row = [fp] + [0.0] * (len(columns) - 1)
                writer.writerow(row)
                print(f"Emotion detection failed: {fp}")
                continue

            row = [fp]
            for col in columns[1:]:
                label = columns_to_label.get(col)
                if label is None or label not in label_to_idx:
                    row.append(0.0)
                else:
                    idx = label_to_idx[label]
                    row.append(float(probs[idx]))

            writer.writerow(row)
            print(f"Processed: {fp}")


def main():
    run_folder_to_csv(input_folder, output_location)


if __name__ == '__main__':
    main()