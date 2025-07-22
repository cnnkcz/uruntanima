from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import os
import cv2
import numpy as np

# FastAPI uygulaması oluştur
app = FastAPI(title="Ürün Tanıma Sistemi")

# Static dosyalar ve templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model ve CSV yükle
MODEL_PATH = r'C:\Users\Lenovo\runs\detect\yolo_detection_train4\weights\best.pt'
CSV_PATH = 'urunler.csv'

model = YOLO(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Resmi oku
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # OpenCV formatına çevir (kutu çizmek için)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Nesne tespiti yap
    results = model(img)
    
    detected_products = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Koordinatları al
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Sınıf ID'sini al
                class_id = int(box.cls[0])
                # Sınıf adını al
                class_name = result.names[class_id]
                # Güven skorunu al
                confidence = float(box.conf[0])
                
                # CSV'de var mı kontrol et
                urun_bilgi = df[df['product_name'] == class_name]
                if not urun_bilgi.empty:
                    urun = urun_bilgi.iloc[0]
                    label = f"{class_name} {confidence:.2f}"
                    detected_products.append({
                        'product_name': urun['product_name'],
                        'price': urun['price'],
                        'description': urun['description'],
                        'confidence': f"{confidence:.2f}"
                    })
                else:
                    label = f"Tanınamadı {confidence:.2f}"
                    detected_products.append({
                        'product_name': "Tanınamadı",
                        'price': "-",
                        'description': "-",
                        'confidence': f"{confidence:.2f}"
                    })

                # Kırmızı kutu çizme
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Etiket arka planı için boyut hesaplama
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_cv, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 0, 255), -1)
                cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # İşaretlenmiş resmi kaydet
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    output_path = os.path.join(static_dir, "detected_image.jpg")
    cv2.imwrite(output_path, img_cv)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result": detected_products,
        "image_url": "/static/detected_image.jpg"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001) 