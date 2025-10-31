import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# --- KONFIGURASI ---
# Sesuaikan ini dengan lokasi file Anda
MODEL_PATH = 'results/best_model.keras'
# Ganti dengan path ke gambar yang ingin Anda uji
GAMBAR_UJI_PATH = 'dataset_padang_food/ayam_goreng/ayam_goreng (1).png' 
IMG_SIZE = (150, 150) # Harus sesuai dengan ukuran saat training

# Daftar label kelas Anda (Ganti sesuai urutan folder di dataset_padang_food)
# Pastikan urutan ini benar, sesuai urutan abjad folder Anda!
CLASS_LABELS = ['Ayam Goreng', 'Ayam Pop', 'Daging Rendang', 'Daging Batokok', 'Gulai Ikan', 'Gulai Tambusu', 'Gulai Tunjang', 'Telur Balado', 'Telur Dadar'] 

# --- 1. MEMUAT MODEL ---
try:
    print("Memuat model terbaik...")
    # Fungsi Keras untuk memuat model dari format .keras
    model = tf.keras.models.load_model(MODEL_PATH) 
    print("✅ Model berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    exit()

# --- 2. MEMUAT DAN MEMPROSES GAMBAR ---
def preprocess_image(img_path, target_size):
    # Memuat gambar dan mengubah ukurannya
    img = image.load_img(img_path, target_size=target_size) 
    # Mengubah gambar ke array numpy
    img_array = image.img_to_array(img)
    # Menambahkan dimensi batch (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    # Normalisasi (seperti saat training)
    img_array = img_array / 255.0 
    return img_array

try:
    processed_image = preprocess_image(GAMBAR_UJI_PATH, IMG_SIZE)
except FileNotFoundError:
    print(f"❌ Error: Gambar tidak ditemukan di path: {GAMBAR_UJI_PATH}")
    exit()

# --- 3. PREDIKSI ---
print("Melakukan prediksi...")
# Memperoleh probabilitas untuk setiap kelas
predictions = model.predict(processed_image) 

# Mendapatkan indeks kelas dengan probabilitas tertinggi
predicted_class_index = np.argmax(predictions[0])
# Mendapatkan nama kelas
predicted_label = CLASS_LABELS[predicted_class_index]
# Mendapatkan probabilitas (kepercayaan)
confidence = predictions[0][predicted_class_index]

# --- 4. OUTPUT HASIL ---
print("\n=== HASIL PREDIKSI ===")
print(f"Gambar: {os.path.basename(GAMBAR_UJI_PATH)}")
print(f"Prediksi Model: {predicted_label}")
print(f"Tingkat Kepercayaan: {confidence*100:.2f}%")
print("======================\n")