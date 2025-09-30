import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import random
import shutil

# -----------------------------
# CONFIGURATION
# -----------------------------
SOURCE_IMAGES = "roboflow/train/images"  # folder containing downloaded images
LABELS_DIR = "roboflow/train/labels"     # folder containing YOLO bounding box labels (.txt)
DATASET_DIR = "dataset"            # output dataset folder
TRAIN_RATIO = 0.8                  # 80% train, 20% valid
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # path to SAM checkpoint

# -----------------------------
# 1. Load SAM model
# -----------------------------
import os
import urllib.request
from segment_anything import sam_model_registry, SamPredictor

# Choose model type (vit_h, vit_l, or vit_b)
sam_model_type = "vit_h"

# Define checkpoint filenames and URLs
sam_checkpoints = {
    "vit_h": {
        "file": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "vit_l": {
        "file": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "vit_b": {
        "file": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
}

SAM_CHECKPOINT = sam_checkpoints[sam_model_type]["file"]
SAM_URL = sam_checkpoints[sam_model_type]["url"]

# Auto-download if checkpoint missing
if not os.path.exists(SAM_CHECKPOINT):
    print(f"Downloading {SAM_CHECKPOINT} ...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
    print("✅ Download complete!")

# Load SAM model
sam = sam_model_registry[sam_model_type](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

print("✅ SAM model loaded successfully!")

# -----------------------------
# 2. Prepare dataset folders
# -----------------------------
for split in ["train", "valid"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "masks", split), exist_ok=True)

# -----------------------------
# 3. Get all images and shuffle
# -----------------------------
image_files = [f for f in os.listdir(SOURCE_IMAGES) if f.lower().endswith((".jpg", ".png"))]
random.shuffle(image_files)
num_train = int(len(image_files) * TRAIN_RATIO)

# -----------------------------
# 4. Process images and generate masks
# -----------------------------
for i, img_file in enumerate(image_files):
    split = "train" if i < num_train else "valid"
    img_path = os.path.join(SOURCE_IMAGES, img_file)
    label_file = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + ".txt")

    # Skip images without labels
    if not os.path.exists(label_file):
        print(label_file)
        print(f"No bounding box label for {img_file}, skipping.")
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    # Read YOLO-format boxes
    with open(label_file, "r") as f:
        boxes = [line.strip().split() for line in f.readlines()]

    # Process each box
    for j, box in enumerate(boxes):
        class_id, x_center, y_center, bw, bh = map(float, box)
        # Convert YOLO format to pixel coordinates
        x1 = int((x_center - bw/2) * w)
        y1 = int((y_center - bh/2) * h)
        x2 = int((x_center + bw/2) * w)
        y2 = int((y_center + bh/2) * h)
        box_prompt = np.array([x1, y1, x2, y2])

        # Generate mask with SAM
        masks, scores, logits = predictor.predict(box=box_prompt, multimask_output=False)
        mask = masks[0].astype(np.uint8) * 255

        # Smooth edges
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        # Save image and mask
        save_img_path = os.path.join(DATASET_DIR, "images", split, f"{os.path.splitext(img_file)[0]}_{j}.png")
        save_mask_path = os.path.join(DATASET_DIR, "masks", split, f"{os.path.splitext(img_file)[0]}_{j}.png")
        cv2.imwrite(save_img_path, img)
        cv2.imwrite(save_mask_path, mask)

        print(f"Saved: {save_img_path} + mask {save_mask_path}")

# -----------------------------
# 5. Create YOLOv8 dataset YAML
# -----------------------------
import yaml

dataset_yaml = {
    "names": ["sunglasses"],
    "nc": 1,
    "train": os.path.join(DATASET_DIR, "images", "train"),
    "valid": os.path.join(DATASET_DIR, "images", "valid"),
    "mask": True
}

yaml_path = os.path.join(DATASET_DIR, "dataset_config.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f)

print(f"YOLOv8-seg dataset YAML created at: {yaml_path}")
