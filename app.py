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
# 0️⃣ KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# =========================================================
st.set_page_config(page_title="Klasifikasi Sampah Otomatis", layout="centered", page_icon="♻️")

# =========================================================
# 1️⃣ SETTINGS: PATH MODEL & LABELS
# =========================================================
MODEL_PATH = "best_tl.h5"            # pastikan file ada di direktori yang sama
CLASS_INDICES_PATH = "class_indices.json"  # optional, sangat disarankan

# Fallback default labels (jika user tidak menyediakan file class_indices.json)
DEFAULT_LABELS = ['Kaca', 'Kardus', 'Kertas', 'Logam', 'Plastik', 'Residu']

# =========================================================
# 2️⃣ LOAD MODEL & CLASS INDICES (CACHE)
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
                d = json.load(f)
            # convert to ordered labels by index
            # d should be like {"Kaca": 0, "Kardus": 1, ...}
            labels = [None] * len(d)
            for k, v in d.items():
                if v < 0:
                    continue
                labels[v] = k
            # filter None if any gap
            labels = [lab for lab in labels if lab is not None]
            return labels
        except Exception as e:
            st.warning(f"Gagal baca {path}: {e}. Menggunakan label default.")
            return DEFAULT_LABELS
    else:
        return DEFAULT_LABELS

# try load (tampilkan pesan yang jelas jika gagal)
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

LABELS = load_class_indices(CLASS_INDICES_PATH)

# =========================================================
# 3️⃣ UTIL: PREPROCESS & PREDIKSI (cached)
# =========================================================
@st.cache_data
def preprocess_image_from_bytes(image_bytes):
    """
    Kembalikan array yang sudah siap untuk diprediksi oleh MobileNetV2 preprocess_input.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # sesuai training TL
    return img_array

def predict_from_array(img_array, top_k=3):
    """
    Mengembalikan label_teratas, confidence, serta dataframe probabilitas semua kelas.
    """
    preds = model.predict(img_array)
    preds = np.squeeze(preds)  # shape (num_classes,)
    # pastikan panjang sama dengan labels
    if preds.shape[0] != len(LABELS):
        st.warning("Jumlah output model tidak sama dengan jumlah LABELS. Periksa class_indices.json dan model.")
    # top-k
    top_idx = np.argsort(preds)[::-1][:top_k]
    top = [(LABELS[i] if i < len(LABELS) else f"kelas_{i}", float(preds[i])) for i in top_idx]
    top_label, top_conf = top[0][0], top[0][1]
    # dataframe untuk chart
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
    "Model dilatih menggunakan MobileNetV2 dan sudah menyesuaikan preprocessing `mobilenet_preprocess`."
)

st.info("Tip: pastikan `best_tl.h5` dan `class_indices.json` (opsional) berada di folder aplikasi saat deploy.")

option = st.radio("Pilih sumber gambar:", ["Upload dari Internal", "Ambil dari Kamera"], horizontal=True)
image_input = None
image_bytes_for_cache = None

if option == "Upload dari Internal":
    uploaded_file = st.file_uploader("Pilih file gambar (JPG / JPEG / PNG). Maks 5MB:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ Ukuran file terlalu besar. Maksimal 5MB.")
        else:
            try:
                image_bytes_for_cache = uploaded_file.read()
                image_input = Image.open(io.BytesIO(image_bytes_for_cache)).convert("RGB")
                st.image(image_input, caption="📁 Gambar yang diunggah", use_container_width=True)
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
            st.image(image_input, caption="📸 Gambar hasil kamera", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal membaca foto: {e}")

# Prediksi
if image_input is not None:
    st.write("---")
    col1, col2 = st.columns([2,3])
    with col1:
        st.markdown("**Preview**")
        st.image(image_input, use_container_width=True)
    with col2:
        st.markdown("**Pengaturan & Output**")
        top_k = st.selectbox("Berapa top-k yang ditampilkan?", [1, 2, 3], index=2)
        if st.button("🔍 Prediksi Sekarang"):
            try:
                with st.spinner("Sedang memproses gambar..."):
                    # preprocess
                    img_arr = preprocess_image_from_bytes(image_bytes_for_cache)
                    # predict
                    label, conf, top_results, df_probs = predict_from_array(img_arr, top_k=top_k)

                    st.success("✅ Prediksi selesai!")
                    # Kartu hasil utama
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

                    # Progress bar (confidence)
                    prog = st.progress(0)
                    # convert conf 0..1 to 0..100
                    prog.progress(int(conf * 100))

                    # Tabel top-k
                    st.markdown("**Hasil Top-k**")
                    top_df = pd.DataFrame(top_results, columns=["Kelas", "Probabilitas"])
                    top_df["Probabilitas (%)"] = (top_df["Probabilitas"] * 100).map("{:.2f}".format)
                    st.table(top_df[["Kelas", "Probabilitas (%)"]])

                    # Chart probabilitas semua kelas
                    st.markdown("**Probabilitas semua kelas**")
                    st.bar_chart(df_probs.set_index("Kelas"))

                    # debug: tunjukkan mapping jika ada perbedaan
                    if len(LABELS) != df_probs.shape[0]:
                        st.warning("Jumlah label berbeda antara class_indices.json dan output model. Periksa class_indices.json.")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
else:
    st.info("Silakan unggah gambar atau ambil foto terlebih dahulu untuk melakukan prediksi.")
