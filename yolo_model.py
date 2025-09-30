from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model
model = YOLO(r"C:\Users\ALOK\PycharmProjects\timepass\yolov8_glasses\glasses_segmentation24\weights\best.pt")

def make_sunnyG(image_path):
    results = model(image_path)

    # Load original image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    for r in results:
        # Get sunglasses class ID
        sunglasses_class_id = [k for k, v in r.names.items() if v == "sunglasses"][0]

        if r.masks is None:
            print("No masks found!")
            continue

        # Filter masks for sunglasses
        mask_indices = (r.boxes.cls.cpu().numpy() == sunglasses_class_id)
        sunglasses_masks = r.masks.data[mask_indices].cpu().numpy()

        print(f"Found {len(sunglasses_masks)} sunglasses")

        for i, m in enumerate(sunglasses_masks):
            # Resize mask back to original image size
            mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask * 255).astype("uint8")

            # --- Erode to shrink mask inward (remove skin at perimeter) ---
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)

            # Apply refined mask
            sunglasses_region = cv2.bitwise_and(img, img, mask=mask)

            # Crop tightly around mask
            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                cropped = sunglasses_region[y_min:y_max, x_min:x_max]

                # Transparent PNG with alpha channel
                bgr = cropped
                alpha = mask[y_min:y_max, x_min:x_max]
                rgba = cv2.merge((bgr, alpha))

                # Ensure results folder exists
                os.makedirs("results", exist_ok=True)

                # Build save path safely
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = f"results/sunglasses_{base_name}.png"

                cv2.imwrite(save_path, rgba)
                cv2.imshow(f"Sunglasses_{base_name}", rgba)
                cv2.destroyWindow(f"Sunglasses_{base_name}")

                return save_path


if __name__ == "__main__":
    print(make_sunnyG("images_test/rohit.png"))
