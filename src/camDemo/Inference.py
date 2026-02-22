import torch
from torchvision import transforms
from src.models.model import build_model
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np

image = Image.open("image path")

device = torch.device("cpu")
mtcnn = MTCNN(image_size = 160, margin = 20, min_face_size = 40, thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True, device = device)
frameCounter = 10
last_emotion = None
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
    labels = ["angry","disgust","fear","happy","sad","surprise"]


def determineEmotion(image):
    face64, box = processImage(image)
    if face64 is None or box is None:
        #print("No face detected")
        return None
    try:
        img = Image.fromarray(face64).convert("L")
    except Exception:
        print("Failed to convert face array to image")
        return None

    transform_local = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_tensor = transform_local(img).unsqueeze(0).float()

    if img_tensor is None:
        print("No face detected")
        return None
    else:
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            predicted_class = int(probs.argmax())
            predicted_label = labels[predicted_class] if labels and len(labels) > predicted_class else str(predicted_class)
        for i, p in enumerate(probs):
            name = labels[i] if i < len(labels) else str(i)
        return (probs.argmax())
    

def processImage(image):
    try:
        img_arr = np.asarray(image.convert('RGB'))
    except Exception:
        img_arr = np.asarray(image)

    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)

    batch = np.expand_dims(img_arr, 0)

    boxes_list, probs_list = mtcnn.detect(batch)

    boxes = boxes_list[0] if boxes_list is not None else None
    probs = probs_list[0] if probs_list is not None else None

    if boxes is None or len(boxes) == 0:
        return np.full((64, 64), 127, dtype=np.uint8), None

    idx = int(np.argmax(probs)) if probs is not None else 0
    x1, y1, x2, y2 = map(int, boxes[idx])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    face = image.crop((x1, y1, x2, y2)).resize((64, 64)).convert("L")
    face64 = np.array(face)

    return face64, (x1, y1, x2, y2)

if(determineEmotion(image) == None):
    print("No face detected")
else:
    print(labels[determineEmotion(image)])