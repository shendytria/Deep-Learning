import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- KONFIGURASI MODEL & LABELS ---
MODEL_PATH = 'results/best_model.keras'
IMG_SIZE = (150, 150)
CLASS_LABELS = [
    'Ayam Goreng', 'Ayam Pop', 'Daging Rendang', 'Daging Batokok',
    'Gulai Ikan', 'Gulai Tambusu', 'Gulai Tunjang',
    'Telur Balado', 'Telur Dadar'
]

# --- FUNGSI CACHE MODEL ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- FUNGSI PRE-PROSES GAMBAR ---
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# --- CSS UNTUK TEMA DARK MODERN ---
st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background-color: #0f0f0f;
        color: #f5f5f5;
    }
    /* Judul */
    h1, h2, h3 {
        color: #f5f5f5;
        font-family: 'Poppins', sans-serif;
    }
    /* Tombol */
    div.stButton > button:first-child {
        background-color: #2d2d2d;
        color: #f5f5f5;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #444;
        border-color: #666;
    }
    /* Card hasil prediksi */
    .prediction-card {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.05);
        margin-top: 1.5rem;
        text-align: center;
    }
    .prediction-label {
        font-size: 1.4rem;
        font-weight: 600;
        color: #00ff88;
    }
    .confidence-label {
        font-size: 1.1rem;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

# --- UI UTAMA ---
st.title("Klasifikasi Makanan Padang")
st.subheader("Prediksi jenis masakan menggunakan model Xception")

model = load_model()

if model:
    uploaded_file = st.file_uploader("Unggah gambar makanan:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_container_width=True)
        
        if st.button("Prediksi"):
            with st.spinner("Model sedang memproses..."):
                processed_image = preprocess_image(uploaded_file)
                predictions = model.predict(processed_image)
                
                predicted_class_index = np.argmax(predictions[0])
                predicted_label = CLASS_LABELS[predicted_class_index]
                confidence = predictions[0][predicted_class_index]

                # Tampilan hasil dalam card
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-label">{predicted_label}</div>
                        <div class="confidence-label">Tingkat Kepercayaan: {confidence*100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
