import streamlit as st

# 🧩 WAJIB: letakkan di sini, paling atas
st.set_page_config(page_title="Klasifikasi Sampah Otomatis", layout="centered")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

# =========================================================
# 1️⃣ LOAD MODEL (cache agar cepat)
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_tl.h5")

model = load_model()

LABELS = ['Kaca', 'Kardus', 'Kertas', 'Logam', 'Plastik', 'Residu']

# =========================================================
# 2️⃣ FUNGSI PREDIKSI
# =========================================================
def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    return LABELS[pred_class], confidence

# =========================================================
# 3️⃣ ANTARMUKA STREAMLIT
# =========================================================
st.title("♻️ Klasifikasi Sampah Otomatis (MobileNetV2)")
st.markdown("Unggah gambar atau ambil foto dari kamera untuk mendeteksi jenis sampah.")

option = st.radio("Pilih sumber gambar:", ["Upload dari Internal", "Ambil dari Kamera"])
image_input = None

if option == "Upload dari Internal":
    uploaded_file = st.file_uploader("Pilih file gambar (jpg/png/jpeg):", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_input = Image.open(uploaded_file)
        st.image(image_input, caption="📁 Gambar yang diunggah", use_container_width=True)

elif option == "Ambil dari Kamera":
    camera_photo = st.camera_input("📸 Ambil foto langsung")
    if camera_photo:
        image_input = Image.open(io.BytesIO(camera_photo.getvalue()))
        st.image(image_input, caption="📸 Gambar hasil kamera", use_container_width=True)

if image_input is not None:
    st.write("---")
    if st.button("🔍 Prediksi Sekarang"):
        with st.spinner("Sedang memproses gambar..."):
            label, conf = predict_image(image_input)
            st.success(f"✅ Prediksi: **{label.upper()}** (Confidence: {conf:.2f})")
else:
    st.info("Silakan unggah gambar atau ambil foto terlebih dahulu untuk melakukan prediksi.")
