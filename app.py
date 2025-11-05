import streamlit as st

# =========================================================
# 0️⃣ KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# =========================================================
st.set_page_config(page_title="Klasifikasi Sampah Otomatis", layout="centered")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

# =========================================================
# 1️⃣ LOAD MODEL (pakai cache agar cepat)
# =========================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_tl.h5")
    return model

model = load_model()

# Label kelas (ubah sesuai dataset kamu)
LABELS = ['Kaca', 'Kardus', 'Kertas', 'Logam', 'Plastik', 'Residu']

# =========================================================
# 2️⃣ FUNGSI PREDIKSI
# =========================================================
def predict_image(img):
    try:
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return LABELS[pred_class], confidence
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        return None, None

# =========================================================
# 3️⃣ ANTARMUKA STREAMLIT
# =========================================================
st.title("♻️ Klasifikasi Sampah Otomatis (MobileNetV2)")
st.markdown(
    "Unggah gambar atau ambil foto dari kamera untuk mendeteksi jenis sampah. "
    "Model ini dilatih menggunakan arsitektur **MobileNetV2**."
)

option = st.radio("Pilih sumber gambar:", ["Upload dari Internal", "Ambil dari Kamera"])
image_input = None

# ---------------------------------------------------------
# UPLOAD GAMBAR DARI INTERNAL
# ---------------------------------------------------------
if option == "Upload dari Internal":
    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG / JPEG / PNG):",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ Ukuran file terlalu besar. Maksimal 5MB.")
        else:
            try:
                image_bytes = uploaded_file.read()
                image_input = Image.open(io.BytesIO(image_bytes)).convert("RGB")
               st.image(image_input, caption="📸 Gambar yang diunggah", use_container_width=True)
            except UnidentifiedImageError:
                st.error("❌ File yang diunggah bukan gambar yang valid.")
            except Exception as e:
                st.warning(f"⚠️ Terjadi peringatan kecil: {e}")

# ---------------------------------------------------------
# FOTO DARI KAMERA
# ---------------------------------------------------------
elif option == "Ambil dari Kamera":
    camera_photo = st.camera_input("📸 Ambil foto langsung")
    if camera_photo is not None:
        try:
            image_input = Image.open(io.BytesIO(camera_photo.getvalue())).convert("RGB")
            st.image(image_input, caption="📸 Gambar hasil kamera", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal membaca foto: {e}")

# ---------------------------------------------------------
# TOMBOL PREDIKSI
# ---------------------------------------------------------
if image_input is not None:
    st.write("---")
    if st.button("🔍 Prediksi Sekarang"):
        with st.spinner("Sedang memproses gambar..."):
            label, conf = predict_image(image_input)
            if label is not None:
                st.success("✅ Prediksi berhasil!")
                st.markdown(
                    f"""
                    <div style='background-color:#E8F5E9;padding:15px;border-radius:10px'>
                        <h3 style='color:#2E7D32;text-align:center'>♻️ {label.upper()}</h3>
                        <p style='text-align:center;font-size:18px'>
                            Tingkat keyakinan: <b>{conf*100:.2f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Tambahkan bar kepercayaan (confidence)
                st.progress(conf)

else:
    st.info("Silakan unggah gambar atau ambil foto terlebih dahulu untuk melakukan prediksi.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow (MobileNetV2)")

