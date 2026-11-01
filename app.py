import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from feature_engineering import create_features

st.set_page_config(page_title="ConvertIQ", page_icon="🛒", layout="wide")
PINK, BLUE, GREY = "#E91E63", "#455A64", "#B0BEC5"
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_resource
def load_model():
    return joblib.load("convertiq_model.joblib")


@st.cache_data
def load_data():
    df = pd.read_csv("convertiq_dashboard_data.csv")
    df["Revenue"] = df["Revenue"].astype(int)
    # split yang sama dengan notebook (80/20, stratified, random_state=42)
    idx, _ = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df["Revenue"])
    return df, df.loc[idx]


cfg = load_model()
model, threshold = cfg["model"], cfg["threshold"]
df, df_eda = load_data()
with open("model_meta.json") as f:
    meta = json.load(f)

st.title("🛒 ConvertIQ: Prediksi Niat Beli Pengunjung E-Commerce")
st.caption("Final Project Learning Path Data ISCOM 2026 | Tema: E-Commerce")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Ringkasan & EDA", "📈 Forecast Order", "🤖 Performa Model", "🎯 Prediksi"])

# ------------------------------------------------------------ TAB 1
with tab1:
    st.caption(f"EDA memakai data training ({len(df_eda):,} sesi, 80% dari {len(df):,} sesi "
               "setelah 125 duplikat dihapus), sama dengan notebook.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sesi (training)", f"{len(df_eda):,}")
    c2.metric("Konversi", f"{df_eda['Revenue'].mean() * 100:.2f}%")
    c3.metric("Pengunjung baru", f"{(df_eda['VisitorType'] == 'New_Visitor').mean() * 100:.1f}%")
    c4.metric("Rata-rata halaman produk", f"{df_eda['ProductRelated'].mean():.1f}")

    a, b = st.columns(2)
    with a:
        m = df_eda.groupby("Month")["Revenue"].mean().reindex(MONTHS) * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(m.index, m.values, color=PINK)
        ax.axhline(df_eda["Revenue"].mean() * 100, ls="--", color=BLUE, label="Rata-rata")
        ax.set_ylabel("Konversi (%)"); ax.set_title("Konversi per bulan"); ax.legend()
        st.pyplot(fig)
    with b:
        v = df_eda.groupby("VisitorType")["Revenue"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(v.index, v.values, color=PINK)
        for i, val in enumerate(v.values):
            ax.text(i, val + 0.4, f"{val:.1f}%", ha="center")
        ax.set_ylabel("Konversi (%)"); ax.set_title("Konversi per tipe pengunjung")
        st.pyplot(fig)

    c, d = st.columns(2)
    for col, ttl, area in [("ProductRelated", "Halaman produk dibuka", c), ("ExitRates", "ExitRates", d)]:
        with area:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.boxplot([df_eda.loc[df_eda.Revenue == 0, col], df_eda.loc[df_eda.Revenue == 1, col]],
                       tick_labels=["Tidak membeli", "Membeli"], showfliers=False)
            ax.set_title(f"{ttl} vs status pembelian")
            st.pyplot(fig)

    with st.expander("Korelasi fitur numerik dengan Revenue"):
        fe = create_features(df_eda.drop(columns=["PageValues"]))
        num = fe.select_dtypes("number").drop(columns=["OperatingSystems", "Browser", "Region", "TrafficType"])
        corr = num.corr()["Revenue"].drop("Revenue").sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(corr.index, corr.values, color=PINK)
        ax.axvline(0, color="black", lw=1); ax.set_xlabel("Korelasi Pearson")
        st.pyplot(fig)

# ------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Forecast jumlah order 3 bulan ke depan")
    hist = pd.read_csv("historical_orders.csv", parse_dates=["Month"])
    fc = pd.read_csv("forecast_results.csv", parse_dates=["Month"])
    mo = pd.read_csv("monthly_orders.csv", parse_dates=["Month"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["Month"], hist["Total_Orders"], marker="o", label="Historis", color=BLUE)
    ax.plot(fc["Month"], fc["Forecast_Orders"], marker="o", ls="--", label="Forecast", color=PINK)
    ax.fill_between(fc["Month"], fc["lower Monthly_Orders"], fc["upper Monthly_Orders"],
                    color=PINK, alpha=0.2, label="Interval kepercayaan")
    ax.set_ylabel("Total order"); ax.legend()
    st.pyplot(fig)
    show = fc.copy()
    show["Month"] = show["Month"].dt.strftime("%b %Y")
    show.columns = ["Bulan", "Forecast", "Batas bawah", "Batas atas"]
    st.dataframe(show.round(0), hide_index=True)
    st.subheader("Completion rate order")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mo["Month"], mo["Completion_Rate"], marker="o", color=BLUE)
    ax.set_ylabel("Completion rate (%)")
    st.pyplot(fig)

# ------------------------------------------------------------ TAB 3
with tab3:
    t = meta["test"]
    st.subheader(f"Model final: {cfg['model_name']}")
    st.write(f"Threshold keputusan **{threshold:.2f}**, parameter terbaik `{meta['best_params']}`. "
             f"Evaluasi pada data test ({meta['n_test']:,} sesi).")
    cols = st.columns(5)
    for col, (lbl, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                      ("Recall", "recall"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(lbl, f"{t[key] * 100:.2f}%")

    st.info("Fitur **PageValues dikeluarkan** dari model karena nilainya dihitung dari halaman yang "
            "dikunjungi sebelum transaksi, sehingga berisiko membocorkan hasil pembelian (data leakage).")

    a, b = st.columns(2)
    with a:
        st.markdown("**Perbandingan model (5-fold CV, data training)**")
        cvt = pd.DataFrame(meta["cv_table"]).T
        cvt.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        st.dataframe(cvt.sort_values("F1", ascending=False).round(4))
    with b:
        st.markdown("**Confusion matrix (data test)**")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        cm = meta["confusion"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["Tidak beli", "Beli"]); ax.set_yticks([0, 1], ["Tidak beli", "Beli"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        st.pyplot(fig)

    c, d = st.columns(2)
    with c:
        st.markdown("**Pemilihan threshold (prediksi CV data training)**")
        thr = pd.DataFrame(meta["thresholds"]).set_index("Threshold")
        st.dataframe(thr.round(4).sort_values("F1", ascending=False))
        st.caption("F1 tertinggi di 0,60, tetapi 0,50 dipilih untuk menjaga recall pembeli.")
    with d:
        st.markdown("**10 fitur dengan koefisien terbesar**")
        coef = pd.Series(meta["top_coef"]).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.barh([n.split("__")[-1] for n in coef.index], coef.values,
                color=[PINK if v > 0 else GREY for v in coef.values])
        ax.axvline(0, color="black", lw=1)
        st.pyplot(fig)

# ------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("Simulasi sesi pengunjung")
    st.write("Isi perilaku pengunjung, lalu tekan tombol untuk melihat peluang ia membeli.")
    c1, c2, c3 = st.columns(3)
    with c1:
        admin = st.number_input("Halaman Administrative", 0, 30, 2)
        admin_d = st.number_input("Durasi Administrative (detik)", 0.0, 3500.0, 60.0)
        info = st.number_input("Halaman Informational", 0, 25, 0)
        info_d = st.number_input("Durasi Informational (detik)", 0.0, 2500.0, 0.0)
        prod = st.number_input("Halaman ProductRelated", 0, 720, 30)
        prod_d = st.number_input("Durasi ProductRelated (detik)", 0.0, 65000.0, 1200.0)
    with c2:
        bounce = st.slider("BounceRates", 0.0, 0.2, 0.01, 0.005)
        exitr = st.slider("ExitRates", 0.0, 0.2, 0.03, 0.005)
        special = st.select_slider("SpecialDay", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)
        weekend = st.checkbox("Kunjungan akhir pekan")
    with c3:
        month = st.selectbox("Bulan", MONTHS, index=MONTHS.index("Nov"))
        visitor = st.selectbox("Tipe pengunjung", ["Returning_Visitor", "New_Visitor", "Other"])
        os_ = st.selectbox("Operating System", sorted(df["OperatingSystems"].unique()))
        browser = st.selectbox("Browser", sorted(df["Browser"].unique()))
        region = st.selectbox("Region", sorted(df["Region"].unique()))
        traffic = st.selectbox("Traffic Type", sorted(df["TrafficType"].unique()))

    if st.button("Prediksi"):
        raw = {
            "Administrative": int(admin), "Administrative_Duration": float(admin_d),
            "Informational": int(info), "Informational_Duration": float(info_d),
            "ProductRelated": int(prod), "ProductRelated_Duration": float(prod_d),
            "BounceRates": float(bounce), "ExitRates": float(exitr), "SpecialDay": float(special),
            "Month": month, "OperatingSystems": int(os_), "Browser": int(browser),
            "Region": int(region), "TrafficType": int(traffic),
            "VisitorType": visitor, "Weekend": bool(weekend),   # harus bool, sama dengan training
        }
        row = pd.DataFrame([raw])[cfg["input_features"]]
        row = create_features(row)[cfg["model_features"]]
        prob = float(model.predict_proba(row)[0, 1])
        if prob >= threshold:
            st.success(f"Berpotensi Membeli. Skor peluang: {prob * 100:.1f}% (threshold {threshold:.2f}).")
            st.write("Rekomendasi: tawarkan voucher atau gratis ongkir agar pembelian terjadi.")
        else:
            st.warning(f"Tidak Berpotensi Membeli. Skor peluang: {prob * 100:.1f}% (threshold {threshold:.2f}).")
            st.write("Rekomendasi: coba retargeting, rekomendasi produk, atau pengingat keranjang.")
        st.progress(min(max(prob, 0.0), 1.0))
