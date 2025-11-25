from ultralytics import YOLO


model_path = 'runs/segment/train7/weights/best.pt'
model = YOLO(model_path)

input_image_path = 'test2.png'
results = model.predict(
    source=input_image_path,
    conf=0.25,
    iou=0.7,
    show=True,
    save=True
)

