import os
import cv2

# paths
root = r"C:\Users\ALOK\PycharmProjects\timepass\roboflow"
mask_train = os.path.join(root, "train", "masks")
mask_valid = os.path.join(root, "valid", "masks")

label_train = os.path.join(root, "train", "labels")
label_valid = os.path.join(root, "valid", "labels")
os.makedirs(label_train, exist_ok=True)
os.makedirs(label_valid, exist_ok=True)

def mask_to_yolo_polygon(mask_path, label_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(label_path, "w") as f:
        for contour in contours:
            if len(contour) < 3:
                continue
            polygon = []
            for point in contour:
                x, y = point[0]
                polygon.append(f"{x/w:.6f} {y/h:.6f}")
            f.write(f"{class_id} " + " ".join(polygon) + "\n")

# convert train masks
for mask_file in os.listdir(mask_train):
    if mask_file.endswith(".png"):
        mask_path = os.path.join(mask_train, mask_file)
        label_path = os.path.join(label_train, mask_file.replace(".png", ".txt"))
        mask_to_yolo_polygon(mask_path, label_path)

# convert valid masks
for mask_file in os.listdir(mask_valid):
    if mask_file.endswith(".png"):
        mask_path = os.path.join(mask_valid, mask_file)
        label_path = os.path.join(label_valid, mask_file.replace(".png", ".txt"))
        mask_to_yolo_polygon(mask_path, label_path)

print("✅ Conversion done: masks → YOLO polygon labels in labels/")
