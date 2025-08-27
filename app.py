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

# === Form input untuk user ===
user_input = {}
for feature in selected_features:
    val = st.text_input(f"{feature}", "0")
    try:
        # Ganti koma dengan titik biar bisa input 0,88 atau 0.88
        val = val.replace(",", ".")
        user_input[feature] = float(val)
    except ValueError:
        st.error(f"Input {feature} harus berupa angka!")
        user_input[feature] = 0.0

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

    # === Tambah grafik Feature Importance ===
    st.subheader("Faktor Paling Berpengaruh (Feature Importance)")
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Fitur": selected_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)  # ascending supaya barh plot rapi

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_imp_df["Fitur"], feat_imp_df["Importance"], color="teal")
    ax.set_xlabel("Tingkat Kepentingan")
    ax.set_title("Feature Importance Random Forest")
    st.pyplot(fig)
