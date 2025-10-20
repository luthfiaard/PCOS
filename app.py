import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# === Load model dan fitur ===
bundle = joblib.load("final_model_with_features.sav")
model = bundle["model"]
selected_features = bundle["features"]

st.title("🧬 Prediksi PCOS dengan Random Forest")
st.write("Masukkan data berikut untuk melakukan prediksi:")
st.write("Jangan gunakan tanda koma (,) ganti dengan tanda titik (.)")

# === Inisialisasi session state untuk riwayat prediksi ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Mapping deskripsi dan contoh rentang input ===
feature_info = {
    "Follicle No. (R)": {"desc": "Masukkan jumlah folikel di ovarium kanan", "range": "Contoh: 0 - 25"},
    "Follicle No. (L)": {"desc": "Masukkan jumlah folikel di ovarium kiri", "range": "Contoh: 0 - 25"},
    "Skin darkening (Y/N)": {"desc": "Apakah terdapat penggelapan kulit", "range": "Pilih: Tidak (0) / Ya (1)"},
    "Weight gain(Y/N)": {"desc": "Apakah terjadi peningkatan berat badan", "range": "Pilih: Tidak (0) / Ya (1)"},
    "Hair growth(Y/N)": {"desc": "Apakah terjadi pertumbuhan rambut berlebih", "range": "Pilih: Tidak (0) / Ya (1)"},
    "Cycle(R/I)": {"desc": "Tipe siklus menstruasi", "range": "Pilih: Regular (0) / Irregular (1)"},
    "AMH(ng/mL)": {"desc": "Masukkan nilai Anti-Müllerian Hormone", "range": "Contoh: 1 - 10"},
    "Cycle length(days)": {"desc": "Masukkan panjang siklus menstruasi dalam hari", "range": "Contoh: 21 - 35"},
    "FSH(mIU/mL)": {"desc": "Masukkan nilai Follicle-Stimulating Hormone", "range": "Contoh: 3 - 15"},
    "LH(mIU/mL)": {"desc": "Masukkan nilai Luteinizing Hormone", "range": "Contoh: 2 - 20"}
}

# === Form input untuk user ===
user_input = {}
for feature in selected_features:
    if feature in feature_info:
        st.markdown(f"**{feature}**  \nℹ️ {feature_info[feature]['desc']} — {feature_info[feature]['range']}")

    if feature in ["Skin darkening (Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)"]:
        pilihan = st.selectbox(feature, ["Tidak (0)", "Ya (1)"], label_visibility="collapsed", key=feature)
        user_input[feature] = 1.0 if "Ya" in pilihan else 0.0

    elif feature == "Cycle(R/I)":
        pilihan = st.selectbox(feature, ["Regular (0)", "Irregular (1)"], label_visibility="collapsed", key=feature)
        user_input[feature] = 1.0 if "Irregular" in pilihan else 0.0

    else:
        val = st.text_input(feature, "", label_visibility="collapsed", key=feature)
        try:
            val = val.replace(",", ".")
            user_input[feature] = float(val) if val.strip() != "" else 0.0
        except ValueError:
            st.error(f"Input {feature} harus berupa angka!")
            user_input[feature] = 0.0

# === Konversi ke DataFrame ===
input_df = pd.DataFrame([user_input], columns=selected_features)

# === Tombol prediksi ===
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    pred_btn = st.button("🔍 Prediksi")
with col2:
    reset_btn = st.button("🔁 Reset Form Input")
with col3:
    history_btn = st.button("📊 Lihat Riwayat Prediksi (jika ada)")

# === Fitur reset form ===
if reset_btn:
    for feature in selected_features:
        if feature in st.session_state:
            del st.session_state[feature]
    st.rerun()

# === Jika tombol prediksi ditekan ===
if pred_btn:
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("📋 Data yang Diuji")

    satuan_map = {
        "Follicle No. (R)": "folikel",
        "Follicle No. (L)": "folikel",
        "AMH(ng/mL)": "ng/mL",
        "Cycle length(days)": "hari",
        "FSH(mIU/mL)": "mIU/mL",
        "LH(mIU/mL)": "mIU/mL"
    }

    for feature, value in user_input.items():
        if feature in ["Skin darkening (Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)"]:
            st.write(f"**{feature}:** {value} (0=Tidak, 1=Ya)")
        elif feature == "Cycle(R/I)":
            st.write(f"**{feature}:** {value} (0=Regular, 1=Irregular)")
        else:
            st.write(f"**{feature}:** {value} {satuan_map.get(feature, '')}")

    # === Hasil prediksi ===
    if prediction == 1:
        st.warning(f"⚠️ Hasil Prediksi: **PCOS** dengan probabilitas {probabilities[1]:.2%}")
        rekomendasi = (
            "Sistem menyarankan untuk melakukan **konsultasi ke dokter spesialis kandungan** "
            "untuk pemeriksaan lebih lanjut."
        )
    else:
        st.success(f"💡 Hasil Prediksi: **Tidak PCOS** dengan probabilitas {probabilities[0]:.2%}")
        rekomendasi = (
            "Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin. "
            "Segera konsultasi ke dokter apabila muncul keluhan lain."
        )

    st.info(f"🧾 **Rekomendasi Sistem:** {rekomendasi}")
    st.caption("⚠️ *Catatan: Sistem ini hanya berfungsi sebagai alat bantu prediksi, bukan diagnosis medis.*")

    # === Simpan ke riwayat ===
    st.session_state.history.append({
        "Prediksi": "PCOS" if prediction == 1 else "Tidak PCOS",
        "Probabilitas_PCOS": f"{probabilities[1]:.2%}",
        "Probabilitas_Tidak_PCOS": f"{probabilities[0]:.2%}",
        **user_input
    })

    # === Visualisasi probabilitas ===
    st.subheader("📊 Visualisasi Probabilitas")
    fig, ax = plt.subplots()
    ax.bar(["Tidak PCOS", "PCOS"], probabilities, color=["skyblue", "salmon"])
    ax.set_ylabel("Probabilitas")
    ax.set_ylim(0, 1)
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10)
    st.pyplot(fig)

# === Fitur lihat riwayat prediksi ===
if history_btn:
    if len(st.session_state.history) > 0:
        st.subheader("📜 Riwayat Prediksi")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("Belum ada riwayat prediksi yang tersimpan.")

