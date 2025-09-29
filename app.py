import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# === Load model + fitur ===
bundle = joblib.load("final_model_with_features.sav")

model = bundle["model"]
selected_features = bundle["features"]

st.title("Prediksi PCOS dengan Random Forest")

st.write("Masukkan data berikut untuk prediksi:")

# === Mapping deskripsi dan contoh rentang input ===
feature_info = {
    "Follicle No. (R)": {
        "desc": "Masukkan jumlah folikel di ovarium kanan",
        "range": "Contoh: 1 - 10"
    },
    "Follicle No. (L)": {
        "desc": "Masukkan jumlah folikel di ovarium kiri",
        "range": "Contoh: 1 - 10"
    },
    "Skin darkening (Y/N)": {
        "desc": "Apakah terdapat penggelapan kulit (0 = Tidak, 1 = Ya)",
        "range": "Contoh: 0 atau 1"
    },
    "Weight gain(Y/N)": {
        "desc": "Apakah terjadi peningkatan berat badan (0 = Tidak, 1 = Ya)",
        "range": "Contoh: 0 atau 1"
    },
    "hair growth(Y/N)": {
        "desc": "Apakah terjadi pertumbuhan rambut berlebih (0 = Tidak, 1 = Ya)",
        "range": "Contoh: 0 atau 1"
    },
    "Cycle(R/I)": {
        "desc": "Tipe siklus menstruasi (0 = Regular, 1 = Irregular)",
        "range": "Contoh: 0 atau 1"
    },
    "AMH(ng/mL)": {
        "desc": "Masukkan nilai Anti-Müllerian Hormone",
        "range": "Contoh: 1 - 10"
    },
    "Cycle length(days)": {
        "desc": "Masukkan panjang siklus menstruasi dalam hari",
        "range": "Contoh: 21 - 35"
    },
    "FSH(mIU/mL)": {
        "desc": "Masukkan nilai Follicle-Stimulating Hormone",
        "range": "Contoh: 3 - 15"
    },
    "LH(mIU/mL)": {
        "desc": "Masukkan nilai Luteinizing Hormone",
        "range": "Contoh: 2 - 20"
    }
}

# === Form input untuk user ===
user_input = {}
for feature in selected_features:
    val = st.text_input(f"{feature}", "")
    try:
        # Ganti koma dengan titik biar bisa input 0,88 atau 0.88
        val = val.replace(",", ".")
        user_input[feature] = float(val) if val.strip() != "" else 0.0
    except ValueError:
        st.error(f"Input {feature} harus berupa angka!")
        user_input[feature] = 0.0

    # Tampilkan deskripsi & contoh di bawah input
    if feature in feature_info:
        st.caption(f"ℹ️ {feature_info[feature]['desc']} — {feature_info[feature]['range']}")

# === Konversi ke DataFrame sesuai urutan fitur ===
input_df = pd.DataFrame([user_input], columns=selected_features)

# === Prediksi saat tombol ditekan ===
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]  # [prob_tidak_PCOS, prob_PCOS]

    if prediction == 1:
        st.warning(f"⚠️ Hasil Prediksi: **PCOS** dengan probabilitas {probabilities[1]:.2%}")

        # === Rule-based Expert System (PCOS) ===
        st.write(
            """
            🧾 **Rekomendasi Sistem (Rule-based Expert System):**  
            Sistem ini menyarankan Anda untuk melakukan **konsultasi lebih lanjut ke dokter spesialis kandungan** untuk pemeriksaan lanjutan.  

            ⚠️ *Catatan:* Sistem ini hanya berfungsi sebagai **alat bantu prediksi**, bukan diagnosis medis.
            """
        )

    else:
        st.success(f"💡 Hasil Prediksi: **Tidak PCOS** dengan probabilitas {probabilities[0]:.2%}")

        # === Rule-based Expert System (Tidak PCOS) ===
        st.write(
            """
            🧾 **Rekomendasi Sistem (Rule-based Expert System):**  
            Tetap jaga pola hidup sehat, lakukan pemeriksaan rutin, dan segera konsultasi ke dokter apabila muncul keluhan lain.  

            ⚠️ *Catatan:* Sistem ini hanya berfungsi sebagai **alat bantu prediksi**, bukan diagnosis medis.
            """
        )

    # === Tambah grafik probabilitas ===
    st.subheader("Visualisasi Probabilitas")
    fig, ax = plt.subplots()
    ax.bar(["Tidak PCOS", "PCOS"], probabilities, color=["skyblue", "salmon"])
    ax.set_ylabel("Probabilitas")
    ax.set_ylim(0, 1)
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10)
    st.pyplot(fig)
