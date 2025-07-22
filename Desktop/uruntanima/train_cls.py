from ultralytics import YOLO
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

model = YOLO('yolov8s.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    auto_augment='randaugment',  # random augmentation
    fliplr=0.7,                  # yatay çevirme olasılığı
    flipud=0.3,                  # dikey çevirme olasılığı
    hsv_h=0.05,                  # renk tonu değişimi
    hsv_s=0.8,                   # doygunluk değişimi
    hsv_v=0.5,                   # parlaklık değişimi
    erasing=0.5,                 # random erasing
    scale=0.7,                   # ölçeklendirme
    translate=0.2,               # çevirme
    degrees=10,                  # döndürme
    val=False                    # Validation split'i kapat, sadece train ve test kullan
)
