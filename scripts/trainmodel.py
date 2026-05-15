from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="yolo_dataset/data.yaml",
    epochs=100,
    imgsz=320,
    batch=4,
    patience=20,
    save=True,
    name="cb_detector"
)