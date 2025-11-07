# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import json
import pandas as pd
import os

# =========================================================
# 0️⃣ KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="Klasifikasi Sampah Otomatis", layout="centered", page_icon="♻️")

# =========================================================
# 1️⃣ PATH MODEL & LABELS
# =========================================================
MODEL_PATH = "best_tl.h5"
CLASS_INDICES_PATH = "class_indices.json"
DEFAULT_LABELS = ['Kaca', 'Kardus', 'Kertas', 'Logam', 'Plastik', 'Residu']

# =========================================================
# 2️⃣ LOAD MODEL & LABELS
# =========================================================
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan di path: {path}")
    model = tf.keras.models.load_model(path)
    return model

@st.cache_data
def load_class_indices(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            labels = [None] * len(data)
            for k, v in data.items():
                if v < 0:
                    continue
                labels[v] = k
            labels = [l for l in labels if l is not None]
            return labels
        except Exception as e:
            st.warning(f"Gagal baca {path}: {e}. Menggunakan label default.")
            return DEFAULT_LABELS
    else:
        return DEFAULT_LABELS

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

LABELS = load_class_indices(CLASS_INDICES_PATH)

# =========================================================
# 3️⃣ PREPROCESS & PREDIKSI
# =========================================================
@st.cache_data
def preprocess_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_from_array(img_array, top_k=3):
    preds = model.predict(img_array)
    preds = np.squeeze(preds)
    if preds.shape[0] != len(LABELS):
        st.warning("Jumlah output model tidak sama dengan jumlah LABELS. Periksa class_indices.json dan model.")
    top_idx = np.argsort(preds)[::-1][:top_k]
    top = [(LABELS[i] if i < len(LABELS) else f"kelas_{i}", float(preds[i])) for i in top_idx]
    top_label, top_conf = top[0][0], top[0][1]
    df = pd.DataFrame({
        "Kelas": [LABELS[i] if i < len(LABELS) else f"kelas_{i}" for i in range(len(preds))],
        "Probabilitas": preds
    }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)
    return top_label, float(top_conf), top, df

# =========================================================
# 4️⃣ STREAMLIT UI
# =========================================================
st.title("♻️ Klasifikasi Sampah Otomatis (MobileNetV2)")
st.markdown(
    "Unggah gambar atau ambil foto langsung dari kamera. "
    "Model menggunakan MobileNetV2 dan preprocessing `mobilenet_preprocess`."
)

st.info("Pastikan `best_tl.h5` dan `class_indices.json` (opsional) tersedia di folder aplikasi.")

option = st.radio("Pilih sumber gambar:", ["Upload dari Internal", "Ambil dari Kamera"], horizontal=True)
image_input = None
image_bytes_for_cache = None

if option == "Upload dari Internal":
    uploaded_file = st.file_uploader("Pilih file gambar (JPG / JPEG / PNG, maks 5MB):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ Ukuran file terlalu besar (maks 5MB).")
        else:
            try:
                image_bytes_for_cache = uploaded_file.read()
                image_input = Image.open(io.BytesIO(image_bytes_for_cache)).convert("RGB")
                st.image(np.array(image_input), caption="📁 Gambar yang diunggah", use_container_width=True)
            except UnidentifiedImageError:
                st.error("❌ File yang diunggah bukan gambar yang valid.")
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan saat membaca file: {e}")

elif option == "Ambil dari Kamera":
    camera_photo = st.camera_input("📸 Ambil foto langsung")
    if camera_photo is not None:
        try:
            image_bytes_for_cache = camera_photo.getvalue()
            image_input = Image.open(io.BytesIO(image_bytes_for_cache)).convert("RGB")
            st.image(np.array(image_input), caption="📸 Gambar hasil kamera", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal membaca foto: {e}")

# =========================================================
# 5️⃣ PROSES PREDIKSI
# =========================================================
if image_input is not None:
    st.write("---")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("**Preview Gambar**")
        try:
            if isinstance(image_input, Image.Image):
                st.image(np.array(image_input), use_container_width=True)
            else:
                st.image(image_input, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Tidak dapat menampilkan gambar: {e}")
            st.stop()

    with col2:
        st.markdown("**Prediksi Model**")
        top_k = st.selectbox("Tampilkan Top-k hasil:", [1, 2, 3], index=2)
        if st.button("🔍 Jalankan Prediksi"):
            try:
                with st.spinner("Sedang memproses..."):
                    img_arr = preprocess_image_from_bytes(image_bytes_for_cache)
                    label, conf, top_results, df_probs = predict_from_array(img_arr, top_k=top_k)
                    st.success("✅ Prediksi selesai!")

                    st.markdown(
                        f"""
                        <div style='background-color:#E8F5E9;padding:12px;border-radius:10px'>
                            <h3 style='color:#2E7D32;text-align:center;margin:6px'>♻️ {label.upper()}</h3>
                            <p style='text-align:center;margin:6px;font-size:16px'>
                                Tingkat keyakinan: <b>{conf*100:.2f}%</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(int(conf * 100))

                    st.markdown("**Hasil Top-k**")
                    top_df = pd.DataFrame(top_results, columns=["Kelas", "Probabilitas"])
                    top_df["Probabilitas (%)"] = (top_df["Probabilitas"] * 100).map("{:.2f}".format)
                    st.table(top_df[["Kelas", "Probabilitas (%)"]])

                    st.markdown("**Probabilitas Semua Kelas**")
                    st.bar_chart(df_probs.set_index("Kelas"))

                    if len(LABELS) != df_probs.shape[0]:
                        st.warning("Jumlah label berbeda antara class_indices.json dan output model.")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
else:
    st.info("Silakan unggah gambar atau ambil foto terlebih dahulu untuk melakukan prediksi.")


