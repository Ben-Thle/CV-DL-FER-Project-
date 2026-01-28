from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN
import torch
from collections import defaultdict, Counter
import random
from pathlib import Path
import pandas as pd
import io
import base64

PipelineDir = Path(__file__).resolve().parent
SrcDir = PipelineDir.parent
ProjectDir = SrcDir.parent
DatasetsDir = SrcDir / "datasets"
ProcessedDir = ProjectDir / "data" / "processed"

device = torch.device("cpu")

mtcnn = MTCNN(image_size = 160, margin = 20, min_face_size = 40, thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True, device = device)

imagecounter = 0
emotioncounter = [0,0,0,0,0,0]


def get_emotionCounter(emotion):
    match emotion:
        case "angry":
            return emotioncounter[0]
        case "disgust":
            return emotioncounter[1]
        case "fear":
            return emotioncounter[2]
        case "happy":
            return emotioncounter[3]
        case "sad":
            return emotioncounter[4]
        case "surprise":
            return emotioncounter[5]

def add_emotionCounter(emotion):
    match emotion:
        case "angry":
            emotioncounter[0]+=1
        case "disgust":
            emotioncounter[1]+=1
        case "fear":
            emotioncounter[2]+=1
        case "happy":
            emotioncounter[3]+=1
        case "sad":
            emotioncounter[4]+=1
        case "surprise":
            emotioncounter[5]+=1

def processImage(image):
    boxes, probs = mtcnn.detect(image)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face = image.crop((x1,y1,x2,y2))
            face64 = face.resize((64,64))
            return face64

emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
splits = ["train", "test"]
datasets = ["FER2013"] #["AffectNet", "FER2013", "RAF-DB"]
samples = []

for dataset in datasets:
    for split in splits:
        for emotion in emotions:
            emotion_dir = DatasetsDir / dataset / split / emotion

            if not emotion_dir.exists():
                continue

            for image_path in sorted(emotion_dir.iterdir()):
                if image_path.suffix.lower() in [".png", ".jpg"]:
                    image = Image.open(image_path)
                    imagecounter += 1
                    add_emotionCounter(emotion)

                    if dataset == "FER2013":
                        image = image.convert("RGB")

                    processed = processImage(image)
                    if processed is not None:
                        image = processed
                    else:
                        image = image.resize((64, 64))

                    image = image.convert("L")

                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    print("Step:", imagecounter,"| Dataset:", dataset,"| Original Split:", split,"| Emotion:", emotion)

                    samples.append({"image_data": img_str, "emotion": emotion, "dataset": dataset})


groups = defaultdict(list)

for s in samples:
    key = (s["dataset"], s["emotion"])
    groups[key].append(s)

train, val, test = [], [], []

random.seed(123)

for group in groups.values():
    random.shuffle(group)
    groupSize = len(group)

    split_train = int(0.7 * groupSize)
    split_val = int(0.15 * groupSize)

    train += group[:split_train]
    val += group[split_train:split_train+split_val]
    test+=group[split_train+split_val:]


#def stats(split):
    #return Counter((s["dataset"], s["emotion"]) for s in split)

#print("Train:", stats(train))
#print("Val:", stats(val))
#print("Test:", stats(test))


base = ProcessedDir / "split_data"

for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
    emotion_groups = defaultdict(list)
    for s in split_data:
        emotion_groups[s["emotion"]].append(s)
    
    for emotion, emotion_data in emotion_groups.items():
        target_dir = base / split_name / emotion
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # emotion_data already contains base64-encoded images from the first loop
        csv_path = target_dir / "data.csv"
        pd.DataFrame(emotion_data).to_csv(csv_path, index=False)
        
        print(f"Saved {len(emotion_data)} processed samples for {split_name}/{emotion} to {csv_path}")