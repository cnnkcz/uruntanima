import os
import random
import shutil

DATASET_DIR = "datasett"
OUTPUT_DIR = "dataset_split"
TRAIN_RATIO = 0.8

# Tüm görselleri ve .txt dosyalarını topla
all_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
all_files.sort()  # İstikrarlı sonuç için
random.shuffle(all_files)

n_train = int(len(all_files) * TRAIN_RATIO)
train_files = all_files[:n_train]
val_files = all_files[n_train:]

# Klasörleri oluştur
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# Dosyaları kopyala
for split, files in [('train', train_files), ('val', val_files)]:
    for img_file in files:
        # Görseli kopyala
        src_img = os.path.join(DATASET_DIR, img_file)
        dst_img = os.path.join(OUTPUT_DIR, split, 'images', img_file)
        shutil.copy2(src_img, dst_img)
        # Etiket dosyasını kopyala
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        src_txt = os.path.join(DATASET_DIR, txt_file)
        dst_txt = os.path.join(OUTPUT_DIR, split, 'labels', txt_file)
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)
        else:
            print(f"Uyarı: {img_file} için .txt dosyası bulunamadı!")

print(f"Toplam {len(all_files)} görsel: {len(train_files)} train, {len(val_files)} val olarak ayrıldı.") 