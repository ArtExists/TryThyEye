import ultralytics
from ultralytics import YOLO




def main():
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data="roboflow/dataset_config.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        project="yolov8_glasses",
        name="glasses_segmentation",
        verbose=True
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # needed for Windows


    main()
