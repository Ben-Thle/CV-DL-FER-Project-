import torch
from torchvision import transforms
from src.models.model import build_model
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np

cap = cv2.VideoCapture(0)

frequency = 10

plt.ion()
fig, ax = plt.subplots()
#fig1, ax1 = plt.subplots()
fig.patch.set_visible(False)
#fig1.patch.set_visible(False)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig.patch.set_visible(False)
#fig1.patch.set_visible(False)

ax.axis("off")
#ax1.axis("off")



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


def determineEmotion():
    global face64
    try:
        img = Image.fromarray(face64).convert("L")
    except Exception:
        print("No face detected")
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
        #print(f"Predicted class: {predicted_class} ({predicted_label})")
        #print full probabilities per label
        #print("Probabilities:")
        for i, p in enumerate(probs):
            name = labels[i] if i < len(labels) else str(i)
            #print(f"  {i} ({name}): {p:.4f}")
        #print(probs)
        return (probs.argmax())
    

def processImage(image):
    boxes, probs = mtcnn.detect(image)
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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    ax.clear()
    ax.imshow(img)

    
    #ax1.clear()
    face64,box = processImage(img)
    if box is not None:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect)
        if last_emotion is not None:
            ax.text(x1, y2 + 50, labels[last_emotion], color="lime")


    ax.axis("off")          
    ax.set_frame_on(False)  

    #ax1.axis("off")          
    #ax1.set_frame_on(False)  

    #ax1.imshow(face64, cmap='gray', vmin=0, vmax=255)
    if frameCounter < frequency:
        frameCounter += 1
    else:
        frameCounter = 0
        if 'box' in locals() and box is not None:
            last_emotion = determineEmotion()
            print("Updated emotion:", labels[last_emotion])
        else:
            print("No face present when trying to determine emotion")

    #print (frameCounter)

    plt.pause(0.001)

cap.release()