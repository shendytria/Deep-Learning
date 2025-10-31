**🍛 Klasifikasi Makanan Padang (Model Xception Transfer Learning)**

Proyek ini bertujuan untuk mengembangkan dan mengimplementasikan model Deep Learning menggunakan arsitektur Xception (dengan teknik Transfer Learning) untuk mengklasifikasikan berbagai jenis makanan tradisional Minangkabau (Padang) berdasarkan gambar.Model yang dihasilkan memiliki akurasi tinggi dan diimplementasikan dalam aplikasi web interaktif menggunakan Streamlit.


**🛠️ Persyaratan Instalasi**

Pastikan Anda memiliki Python 3.x terinstal. 

Buat dan aktifkan Virtual Environment sebelum instalasi.

Instalasi Dependencies:Semua pustaka yang diperlukan tercantum dalam requirements.txt.

install -r requirements.txt


**📂 Struktur Proyek**

deep_learning/

├── dataset_padang_food/

│   ├── (dst. 9 kelas)

├── results/

│   ├── best_model.keras  <-- Model terbaik yang sudah dilatih

│   └── evaluation_report.txt

├── model_training.py     <-- Script untuk melatih dan mengevaluasi model

├── app.py                <-- Script aplikasi web Streamlit (UI/UX)

├── requirements.txt      <-- Daftar semua pustaka

└── README.md


**⚙️ Cara Menjalankan**
1. Pelatihan Model (Generating Model)
  
   Jalankan script pelatihan untuk membuat file best_model.keras dan laporan evaluasi di folder results/

   python model_training.py
2. Demonstrasi UI/UX (Streamlit App)

   Gunakan script app.py untuk menjalankan aplikasi web interaktif yang memuat model yang telah dilatih

   streamlit run app.py
   Aplikasi akan terbuka otomatis di browser Anda (biasanya di http://localhost:8501).
