import cv2
import time
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from src.models.model import build_model
from src.explainability.gradcam import GradCAM, overlay_red, load_checkpoint_flexible
from facenet_pytorch import MTCNN
from collections import Counter

input = cv2.VideoCapture("input_location")

device = torch.device("cpu")
mtcnn = MTCNN(image_size = 160, margin = 20, min_face_size = 40, thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True, device = device)
checkpoint = torch.load("src/camDemo/checkpoint_epoch_61.pt", map_location='cpu')


src_fps = input.get(cv2.CAP_PROP_FPS) or 25.0
frame_duration = 1.0 / src_fps
counter = []
framecounter = 0
images = []
labels = ["angry","disgust","fear","happy","sad","surprise"]


state_dict = None
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('model')
else:
    state_dict = checkpoint
num_classes = 6
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

target_layer = model.layer4[-1]

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
    framecounter+=1
    t0 = time.perf_counter()
    ret, frame = input.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    emo_idx = determineEmotion(pil)
    if emo_idx != None:
        counter.append(emo_idx)

    if emo_idx != None:
        print("Frame", framecounter,":",  labels[emo_idx])
    else:
        print("Frame", framecounter,": No face detected")

    face64, bbox = processImage(pil)
    gradcam = GradCAM(model, target_layer)
    x = torch.tensor(face64).unsqueeze(0).unsqueeze(0).float().to(device)
    result = gradcam(x)

    gradcam.close()

    cam = result.cam

    saved_frame = frame.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = max(0, int(x2))
        y2 = max(0, int(y2))
        if x2 > x1 and y2 > y1:
            try:
                face_pil = Image.fromarray(face64).convert("RGB")
                overlay_face_pil = overlay_red(face_pil, cam)
                overlay_face_np = cv2.cvtColor(np.array(overlay_face_pil), cv2.COLOR_RGB2BGR)

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                overlay_resized = cv2.resize(overlay_face_np, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)

                roi = saved_frame[y1:y2, x1:x2]
                if roi.shape[:2] == overlay_resized.shape[:2]:
                    alpha = 0.7
                    blended = cv2.addWeighted(overlay_resized, alpha, roi, 1 - alpha, 0)
                    saved_frame[y1:y2, x1:x2] = blended
                else:
                    saved_frame[y1:y2, x1:x2] = overlay_resized[0:roi.shape[0], 0:roi.shape[1]]
            except Exception as e:
                print("Overlay paste failed:", e)

    images.append(saved_frame)
    #cv2.imshow('frame', saved_frame)

    elapsed = time.perf_counter() - t0
    wait = frame_duration - elapsed
    if wait > 0:
        time.sleep(wait)
        cv2.waitKey(1)
    else:
        cv2.waitKey(1)

input.release()
cv2.destroyAllWindows()
c = Counter(counter)
print("Predicted emotion:", labels[c.most_common(1)[0][0]])

def generate_video():
    video_name = 'gradCamVideo.avi'

    if len(images) == 0:
        print("No frames available, skipping video generation.")
        return

    frame = images[0]
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), src_fps, (width, height))

    for image in images:
        video.write(image)

    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

generate_video()