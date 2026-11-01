# ConvertIQ: Prediksi Niat Beli Pengunjung E-Commerce

Final Project **Learning Path Data ISCOM 2026** | Tema: **E-Commerce**

ConvertIQ adalah aplikasi web (Streamlit) yang memprediksi apakah seorang pengunjung toko online **akan melakukan pembelian atau tidak** berdasarkan perilakunya selama sesi browsing. Hasilnya membantu tim bisnis menentukan pengunjung mana yang perlu diberi penawaran (voucher, gratis ongkir, retargeting) agar konversi meningkat.

🔗 **Aplikasi live:** https://urlcovertiq-isqom.streamlit.app/
📓 **Notebook Colab:** https://colab.research.google.com/drive/1h1L3hrNp8TRogKNbf9nE42CcJQeOklc8?usp=sharing

---

## 📌 Latar Belakang

Pada e-commerce, mayoritas pengunjung hanya melihat-lihat tanpa membeli. Pada dataset ini hanya sekitar **15,6%** sesi yang berakhir dengan pembelian. Jika perilaku pengunjung bisa dibaca lebih awal, tim marketing dapat fokus pada sesi yang berpotensi membeli dan mengintervensi sesi yang hampir meninggalkan situs.

## 📊 Dataset

| Item | Keterangan |
|---|---|
| Sumber | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset? |
| Berkas | `convertiq_dashboard_data.csv` |
| Jumlah baris | 12.205 sesi |
| Jumlah kolom | 18 |
| Target | `Revenue` (True = membeli, False = tidak membeli) |
| Distribusi target | ±15,6% membeli, ±84,4% tidak membeli (tidak seimbang) |

**Fitur utama**

| Kelompok | Kolom |
|---|---|
| Halaman dikunjungi | `Administrative`, `Informational`, `ProductRelated` |
| Durasi per jenis halaman | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` |
| Metrik perilaku | `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| Konteks sesi | `Month`, `VisitorType`, `Weekend` |
| Teknis | `OperatingSystems`, `Browser`, `Region`, `TrafficType` |

**Fitur hasil rekayasa** (`feature_engineering.py`)

| Fitur | Rumus |
|---|---|
| `TotalPages` | Administrative + Informational + ProductRelated |
| `TotalDuration` | Jumlah ketiga durasi halaman |
| `ProductPageRatio` | ProductRelated / TotalPages |
| `AvgProductDuration` | ProductRelated_Duration / ProductRelated |

## 🔄 Alur Pengerjaan

`Data Ingestion → Preprocessing & Feature Engineering → EDA → Modeling → Evaluasi → Forecasting Order → Deployment (Streamlit)`

## 🤖 Model

Data dibagi 80% training dan 20% test dengan **stratified split** karena kelas tidak seimbang. Preprocessing memakai `StandardScaler` untuk fitur numerik dan `OneHotEncoder` untuk fitur kategorikal, dibungkus dalam satu `Pipeline`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,844 | 0,501 | 0,806 | 0,618 | 0,911 |
| Random Forest | 0,896 | 0,751 | 0,497 | 0,598 | 0,923 |
| Gradient Boosting | 0,900 | 0,705 | 0,626 | 0,663 | 0,936 |
| **Gradient Boosting (tuned)** | **0,902** | **0,688** | **0,691** | **0,689** | **0,937** |

- **Model terpilih:** Gradient Boosting hasil tuning (`GridSearchCV`, 5-fold, metrik ROC-AUC).
- **Parameter terbaik:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`.
- **Threshold keputusan:** 0,43 (dipilih dari F1 terbaik).
- Karena data tidak seimbang, akurasi saja tidak cukup. Evaluasi menitikberatkan pada **F1-score, recall, dan ROC-AUC**.

**Keterbatasan:** threshold dipilih berdasarkan data test sehingga hasilnya sedikit optimistis. Untuk produksi sebaiknya dipilih lewat validation set terpisah.

## 📈 Forecasting Order

Folder `Forecasting/` berisi data historis order bulanan, model ARIMA, dan hasil forecast 3 bulan ke depan (Des 2025 sampai Feb 2026) beserta interval kepercayaannya. Hasilnya ditampilkan pada tab **Forecast Order** di aplikasi.

## 🖥️ Fitur Aplikasi

| Tab | Isi |
|---|---|
| 📊 Ringkasan & EDA | KPI utama, konversi per bulan, konversi per tipe pengunjung, korelasi fitur |
| 📈 Forecast Order | Tren order historis, forecast 3 bulan, completion rate |
| 🤖 Performa Model | Perbandingan model dan 15 fitur terpenting |
| 🎯 Prediksi | Form input sesi pengunjung, hasil peluang membeli, dan rekomendasi aksi |

## 📁 Struktur Repository

```
ConvertIQ/
├── app.py                          # aplikasi Streamlit
├── train_model.py                  # melatih dan menyimpan model
├── feature_engineering.py          # fungsi pembuatan fitur baru
├── requirements.txt                # daftar library
├── model.pkl                       # model hasil training (dipakai app.py)
├── model_meta.json                 # threshold, parameter, dan hasil evaluasi
├── feature_importance.csv          # fitur terpenting
├── convertiq_dashboard_data.csv    # dataset utama
├── monthly_orders.csv              # data order bulanan
├── historical_orders.csv           # data historis order
├── forecast_results.csv            # hasil forecast
├── convertiq_model.joblib          # berkas model dari eksperimen sebelumnya
└── Forecasting/                    # data dan model ARIMA
```

## 🚀 Menjalankan Secara Lokal

Gunakan **Python 3.11 atau 3.12** (Python 3.15 belum didukung scikit-learn 1.6.1).

```bash
# 1. Clone repository
git clone https://github.com/alyadarwanti1201-hub/ConvertIQ.git
cd ConvertIQ

# 2. Buat dan aktifkan virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install library
pip install -r requirements.txt

# 4. (Opsional) latih ulang model
python train_model.py

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`.

## 🧰 Teknologi

Python, pandas, NumPy, scikit-learn 1.6.1, matplotlib, joblib, Streamlit.

## 👥 Tim

| Nama |
|---|
|Raditya Aji Respati Juminar|
|Najwa Aulia Syahrani|
|Alya Darmawanti Armunanto|
|Nagita Nur Febrina|  

---

_Dibuat untuk Learning Path Data ISCOM 2026._
