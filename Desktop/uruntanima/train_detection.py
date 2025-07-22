from ultralytics import YOLO

# Nesne tespiti için YOLOv8 modeli başlat
model = YOLO('yolov8s.pt')  # detection için yolov8s kullan

# Eğitim - data.yaml dosyasını kullan
model.train(
    data='data.yaml',  # data.yaml dosyasını kullan
    epochs=50,  # 100 yerine 50 epoch
    imgsz=640,
    batch=16,
    name='yolo_detection_train'
)