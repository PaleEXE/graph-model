from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 segmentation model (small size)
model = YOLO('runs/segment/train7/weights/best.pt')

# 2. Start Training!
results = model.val(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    device=0 # Use GPU 0
)

