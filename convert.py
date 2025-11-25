from ultralytics.data.converter import convert_coco

# Define the directory where your COCO JSON files are located
coco_labels_dir = 'yolo_dataset/annotations/'

# Define the directory where the new YOLO-format labels will be saved
yolo_save_dir = 'yolo_dataset/'

# Run the conversion for segmentation (use_segments=True)
convert_coco(
    labels_dir=coco_labels_dir,
    save_dir=yolo_save_dir,
    use_segments=True,
)
