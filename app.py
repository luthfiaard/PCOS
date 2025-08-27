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
    # step=0.01 agar bisa ketik manual angka desimal, format="%.2f" untuk valid input
    user_input[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.2f")

# === Konversi ke DataFrame sesuai urutan fitur ===
input_df = pd.DataFrame([user_input], columns=selected_features)

# === Prediksi saat tombol ditekan ===
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]  # [prob_tidak_PCOS, prob_PCOS]

    if prediction == 1:
        st.success(f"💡 Hasil Prediksi: **PCOS** dengan probabilitas {probabilities[1]:.2%}")
    else:
        st.info(f"💡 Hasil Prediksi: **Tidak PCOS** dengan probabilitas {probabilities[0]:.2%}")

    # === Tambah grafik probabilitas ===
    st.subheader("Visualisasi Probabilitas")
    fig, ax = plt.subplots()
    ax.bar(["Tidak PCOS", "PCOS"], probabilities, color=["skyblue", "salmon"])
    ax.set_ylabel("Probabilitas")
    ax.set_ylim(0, 1)
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10)
    st.pyplot(fig)
