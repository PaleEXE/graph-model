from ultralytics import YOLO

model_path = "runs/segment/train6/weights/best.pt"
model = YOLO(model_path)

input_image_path = "test4.DNG"
results = model.predict(
    source=input_image_path, conf=0.40, iou=0.7, show=True, save=True
)
