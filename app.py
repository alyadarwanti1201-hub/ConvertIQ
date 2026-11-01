import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from feature_engineering import create_features

st.set_page_config(page_title="ConvertIQ", page_icon="🛒", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("convertiq_dashboard_data.csv")
    df["Revenue"] = df["Revenue"].astype(int)
    return df


model = load_model()
df = load_data()
with open("model_meta.json") as f:
    meta = json.load(f)
threshold = meta["threshold"]

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.title("🛒 ConvertIQ: Prediksi Niat Beli Pengunjung E-Commerce")
st.caption("Final Project Learning Path Data ISCOM 2026 | Tema: E-Commerce")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Ringkasan & EDA", "📈 Forecast Order", "🤖 Performa Model", "🎯 Prediksi"]
)

# ---------------------------------------------------------------- TAB 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah sesi", f"{len(df):,}")
    c2.metric("Konversi (Revenue)", f"{df['Revenue'].mean() * 100:.1f}%")
    c3.metric("Pengunjung baru", f"{(df['VisitorType'] == 'New_Visitor').mean() * 100:.1f}%")
    c4.metric("Rata-rata PageValues", f"{df['PageValues'].mean():.2f}")

    col_a, col_b = st.columns(2)

    with col_a:
        conv_m = df.groupby("Month")["Revenue"].mean().reindex(MONTH_ORDER) * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(conv_m.index, conv_m.values, color="#2E75B6")
        ax.set_ylabel("Konversi (%)")
        ax.set_title("Tingkat konversi per bulan")
        st.pyplot(fig)

    with col_b:
        conv_v = df.groupby("VisitorType")["Revenue"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(conv_v.index, conv_v.values, color="#ED7D31")
        ax.set_ylabel("Konversi (%)")
        ax.set_title("Konversi berdasarkan tipe pengunjung")
        st.pyplot(fig)

    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        data_pv = df[df["PageValues"] > 0]
        ax.hist([data_pv.loc[data_pv.Revenue == 0, "PageValues"],
                 data_pv.loc[data_pv.Revenue == 1, "PageValues"]],
                bins=30, label=["Tidak beli", "Beli"], color=["#A6A6A6", "#2E75B6"])
        ax.set_xlabel("PageValues (hanya > 0)")
        ax.set_title("PageValues vs keputusan beli")
        ax.legend()
        st.pyplot(fig)
    with col_d:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.boxplot([df.loc[df.Revenue == 0, "ExitRates"], df.loc[df.Revenue == 1, "ExitRates"]],
                   tick_labels=["Tidak beli", "Beli"])
        ax.set_title("ExitRates vs keputusan beli")
        st.pyplot(fig)

    with st.expander("Lihat korelasi fitur numerik dengan Revenue"):
        fe = create_features(df)
        num = fe.select_dtypes("number").drop(columns=["OperatingSystems", "Browser", "Region", "TrafficType"])
        corr = num.corr()["Revenue"].drop("Revenue").sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(corr.index, corr.values, color="#2E75B6")
        ax.set_xlabel("Korelasi dengan Revenue")
        st.pyplot(fig)

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Forecast jumlah order 3 bulan ke depan")
    hist = pd.read_csv("historical_orders.csv", parse_dates=["Month"])
    fc = pd.read_csv("forecast_results.csv", parse_dates=["Month"])
    mo = pd.read_csv("monthly_orders.csv", parse_dates=["Month"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["Month"], hist["Total_Orders"], marker="o", label="Historis", color="#2E75B6")
    ax.plot(fc["Month"], fc["Forecast_Orders"], marker="o", linestyle="--", label="Forecast", color="#ED7D31")
    ax.fill_between(fc["Month"], fc["lower Monthly_Orders"], fc["upper Monthly_Orders"],
                    color="#ED7D31", alpha=0.2, label="Interval kepercayaan")
    ax.set_ylabel("Total order")
    ax.legend()
    st.pyplot(fig)

    show = fc.copy()
    show["Month"] = show["Month"].dt.strftime("%b %Y")
    show.columns = ["Bulan", "Forecast", "Batas bawah", "Batas atas"]
    st.dataframe(show.round(0), hide_index=True)

    st.subheader("Completion rate order")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mo["Month"], mo["Completion_Rate"], marker="o", color="#548235")
    ax.set_ylabel("Completion rate (%)")
    st.pyplot(fig)
    st.caption("Catatan: completion rate Nov 2025 turun tajam (±69%), kemungkinan karena order masih "
               "berstatus diproses saat data diambil.")

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Perbandingan model (data test, 20%)")
    res = pd.DataFrame(meta["results"]).T.round(3)
    st.dataframe(res)
    st.write(f"Model terpilih: **Gradient Boosting (tuned)** dengan parameter terbaik "
             f"`{meta['best_params']}` dan threshold keputusan **{threshold:.2f}**.")
    st.info("Data tidak seimbang (±16% pembeli), jadi akurasi saja menyesatkan. "
            "Kami menilai model dengan F1-score, recall, dan ROC-AUC.")
    imp = pd.read_csv("feature_importance.csv", index_col=0)["importance"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#2E75B6")
    ax.set_title("15 fitur terpenting")
    st.pyplot(fig)

# ---------------------------------------------------------------- TAB 4
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
        pagev = st.number_input("PageValues", 0.0, 400.0, 10.0)
        special = st.select_slider("SpecialDay", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)
        weekend = st.checkbox("Kunjungan akhir pekan")
    with c3:
        month = st.selectbox("Bulan", MONTH_ORDER, index=MONTH_ORDER.index("Nov"))
        visitor = st.selectbox("Tipe pengunjung", ["Returning_Visitor", "New_Visitor", "Other"])
        os_ = st.selectbox("Operating System", sorted(df["OperatingSystems"].unique()))
        browser = st.selectbox("Browser", sorted(df["Browser"].unique()))
        region = st.selectbox("Region", sorted(df["Region"].unique()))
        traffic = st.selectbox("Traffic Type", sorted(df["TrafficType"].unique()))

    if st.button("Prediksi"):
        row = pd.DataFrame([{
            "Administrative": admin, "Administrative_Duration": admin_d,
            "Informational": info, "Informational_Duration": info_d,
            "ProductRelated": prod, "ProductRelated_Duration": prod_d,
            "BounceRates": bounce, "ExitRates": exitr, "PageValues": pagev,
            "SpecialDay": special, "Month": month, "OperatingSystems": os_,
            "Browser": browser, "Region": region, "TrafficType": traffic,
            "VisitorType": visitor, "Weekend": int(weekend),
        }])
        row = create_features(row)[meta["features"]]
        prob = model.predict_proba(row)[0, 1]

        if prob >= threshold:
            st.success(f"Peluang membeli: {prob * 100:.1f}%. Pengunjung ini berpotensi melakukan pembelian.")
            st.write("Rekomendasi: tampilkan penawaran (voucher, gratis ongkir) agar konversi terjaga.")
        else:
            st.warning(f"Peluang membeli: {prob * 100:.1f}%. Pengunjung ini kemungkinan tidak membeli.")
            st.write("Rekomendasi: pertimbangkan retargeting, rekomendasi produk, atau pengingat keranjang.")
        st.progress(float(prob))
