import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
# Perbaikan: Impor metrics secara namespace
from sklearn import metrics 
from sklearn.metrics import classification_report, roc_auc_score 
import json
import matplotlib.pyplot as plt 
import seaborn as sns

# ====== KONFIGURASI DASAR ======
IMG_SIZE = (150, 150)  # Pertahankan ukuran kecil untuk memori
BATCH_SIZE = 8
EPOCHS = 20  # Tetap 20, tapi akan dihentikan oleh EarlyStopping
DATASET_DIR = 'dataset_padang_food'
RESULTS_DIR = 'results' # Ubah nama folder hasil agar tidak menimpa yang lama

os.makedirs(RESULTS_DIR, exist_ok=True)

# ====== DATA AUGMENTATION (TIDAK BERUBAH) ======
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='training')
val_gen = train_datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='validation')
test_gen = test_datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

CLASS_LABELS = sorted(os.listdir(DATASET_DIR))

# ====== MODEL XCEPTION (PENAMBAHAN DROPOUT) ======
base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # <--- MENAMBAHKAN DROPOUT untuk MENGATASI OVERFITTING
x = Dense(128, activation='relu')(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ====== CALLBACKS BARU (ModelCheckpoint & EarlyStopping) ======
# 1. Simpan model terbaik dalam format .keras
checkpoint = ModelCheckpoint(
    os.path.join(RESULTS_DIR, 'best_model.keras'), 
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 2. Hentikan training jika val_accuracy tidak membaik selama 5 epoch
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    mode='max',
    verbose=1,
    restore_best_weights=True # Menggunakan bobot terbaik saat dihentikan
)

# ====== TRAINING ======
start_train = time.time()
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stopping] # Menggunakan kedua callbacks
)
training_time = time.time() - start_train

# ====== FUNGSI VISUALISASI GRAFIK ACCURACY & LOSS ======
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # --- Subplot 1: Model Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # --- Subplot 2: Model Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, 'training_plot.png')
    plt.savefig(plot_path)
    print(f"Grafik training tersimpan di: {plot_path}")


# ====== FUNGSI PLOTTING CONFUSION MATRIX ======
def plot_confusion_matrix(y_true, y_pred_classes, class_labels):
    # Hitung Confusion Matrix data mentah
    cm = metrics.confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(12, 10))
    # Buat heatmap menggunakan Seaborn
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',    
        cmap='Blues', 
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Simpan grafik
    plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion Matrix tersimpan di: {plot_path}")


# Panggil fungsi plot setelah training selesai
plot_history(history)

# ====== TESTING & EVALUASI (Load model terbaik yang disimpan) ======
# Load model terbaik sebelum testing untuk hasil yang paling akurat
model.load_weights(os.path.join(RESULTS_DIR, 'best_model.keras'))
start_test = time.time()
test_gen.reset()
y_pred = model.predict(test_gen)
testing_time = time.time() - start_test

y_true = test_gen.classes
y_pred_classes = np.argmax(y_pred, axis=1)

# Panggil fungsi Confusion Matrix
plot_confusion_matrix(y_true, y_pred_classes, CLASS_LABELS)

report = classification_report(y_true, y_pred_classes, output_dict=True)
roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')

# ====== SIMPAN LAPORAN ======
with open(os.path.join(RESULTS_DIR, 'evaluation_report.txt'), 'w') as f:
    f.write("=== EVALUATION REPORT (XCEPTION with DROPOUT & ES) ===\n\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Testing Time: {testing_time:.2f} seconds\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(json.dumps(report, indent=4))

print("\n✅ Training & evaluasi selesai!")
print(f"🧾 Laporan tersimpan di: {os.path.join(RESULTS_DIR, 'evaluation_report.txt')}")
print(f"💾 Model terbaik tersimpan di: {os.path.join(RESULTS_DIR, 'best_model.keras')}")
print(f"🖼️ Grafik Training tersimpan di: {os.path.join(RESULTS_DIR, 'training_plot.png')}")
print(f"📊 Confusion Matrix tersimpan di: {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")
