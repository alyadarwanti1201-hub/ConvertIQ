# 🛒 ConvertIQ: Prediksi Niat Beli Pengunjung E-Commerce

Final Project **Learning Path Data ISCOM 2026** | Tema: **E-Commerce**

ConvertIQ adalah aplikasi web (Streamlit) yang memprediksi apakah seorang pengunjung toko online **akan melakukan pembelian atau tidak** berdasarkan perilakunya selama sesi browsing. Hasilnya membantu tim bisnis menentukan pengunjung mana yang perlu diberi penawaran (voucher, gratis ongkir, retargeting) agar konversi meningkat.

🔗 **Aplikasi live:** _(https://urlcovertiq-isqom.streamlit.app/)_
📓 **Notebook Colab:** _(https://colab.research.google.com/drive/1h1L3hrNp8TRogKNbf9nE42CcJQeOklc8?usp=sharing)_

---

## 📌 Latar Belakang

Pada e-commerce, mayoritas pengunjung hanya melihat-lihat tanpa membeli. Pada dataset ini hanya sekitar **15,6%** sesi yang berakhir dengan pembelian. Jika perilaku pengunjung bisa dibaca lebih awal, tim marketing dapat fokus pada sesi yang berpotensi membeli dan mengintervensi sesi yang hampir meninggalkan situs.

## 📊 Dataset

| Item | Keterangan |
|---|---|
| Sumber | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset? (`online_shoppers_intention.csv`) |
| Data mentah | 12.330 baris, 18 kolom |
| Setelah pembersihan | **12.205 baris** (125 baris duplikat dihapus), berkas `convertiq_dashboard_data.csv` |
| Missing value | Tidak ada |
| Target | `Revenue` (True = membeli, False = tidak membeli) |
| Distribusi target | 15,63% membeli (1.908 sesi), 84,37% tidak membeli (10.297 sesi), data tidak seimbang |

**Fitur input model (16 kolom)**

| Kelompok | Kolom |
|---|---|
| Halaman dikunjungi | `Administrative`, `Informational`, `ProductRelated` |
| Durasi per jenis halaman | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` |
| Metrik perilaku | `BounceRates`, `ExitRates`, `SpecialDay` |
| Konteks sesi | `Month`, `VisitorType`, `Weekend` |
| Teknis | `OperatingSystems`, `Browser`, `Region`, `TrafficType` |

> **`PageValues` sengaja dikeluarkan dari model.** Nilainya dihitung dari halaman yang dikunjungi sebelum transaksi, sehingga mengandung informasi hasil pembelian dan berisiko menyebabkan *data leakage*. Pada data ini sesi dengan `PageValues = 0` hanya 3,9% yang membeli, sedangkan `PageValues > 0` mencapai 56,3%.

**Fitur hasil rekayasa** (`feature_engineering.py`), sehingga total model memakai **20 fitur**

| Fitur | Rumus |
|---|---|
| `TotalPages` | Administrative + Informational + ProductRelated |
| `TotalDuration` | Jumlah ketiga durasi halaman |
| `ProductPageRatio` | ProductRelated / TotalPages |
| `AvgProductDuration` | ProductRelated_Duration / ProductRelated |

## 🔄 Alur Pengerjaan

`Data Ingestion → Preprocessing & Feature Engineering → EDA → Modeling → Evaluasi → Forecasting Order → Deployment (Streamlit)`

## 🔍 Temuan EDA (data training)

EDA dilakukan hanya pada data training (9.764 sesi) agar tidak bocor ke data test.

- Konversi tertinggi terjadi pada **November (25,4%)**, terendah pada Februari (2,1%).
- **New Visitor** berkonversi 24,09%, jauh di atas Returning Visitor (14,20%).
- Korelasi terkuat dengan `Revenue`: `ExitRates` (-0,205, negatif), lalu `TotalPages` (+0,153) dan `ProductRelated` (+0,148).
- Pengunjung yang membeli rata-rata membuka 47 halaman produk, dibanding 29 pada yang tidak membeli.

## 🤖 Model

Data dibagi 80% training (9.764 sesi) dan 20% test (2.441 sesi) dengan **stratified split** (`random_state=42`). Preprocessing memakai `StandardScaler` untuk fitur numerik dan `OneHotEncoder` untuk fitur kategorikal, dibungkus dalam `Pipeline`.

**Perbandingan model (5-fold cross-validation, data training)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,8432 | 0,4788 | 0,0321 | 0,0601 | 0,7439 |
| Decision Tree | 0,7642 | 0,2606 | 0,2752 | 0,2673 | 0,5650 |
| Random Forest | 0,8465 | 0,5641 | 0,0747 | 0,1317 | 0,7616 |
| **Logistic Regression Balanced** | 0,6420 | 0,2663 | 0,7339 | **0,3907** | 0,7436 |
| Decision Tree Balanced | 0,7744 | 0,2828 | 0,2910 | 0,2867 | 0,5775 |
| Random Forest Balanced | 0,8465 | 0,5718 | 0,0734 | 0,1298 | 0,7701 |

- **Model final:** Logistic Regression dengan `class_weight='balanced'`, dipilih dari F1 terbaik. GridSearchCV pada parameter `C` memilih `C=1`.
- **Threshold keputusan:** 0,50, dipilih dari prediksi cross-validation data training. F1 tertinggi sebenarnya di 0,60 (0,3921), tetapi selisihnya sangat kecil dari 0,50 (0,3905) sehingga 0,50 dipilih untuk menjaga recall pembeli.

**Hasil pada data test**

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| 64,77% | 27,75% | 78,01% | 40,93% | 75,59% |

Confusion matrix: 1.283 benar-negatif, 776 salah-positif, 84 salah-negatif, 298 benar-positif. Model berhasil menangkap 78% dari seluruh pembeli, dengan konsekuensi banyak pengunjung yang diprediksi membeli ternyata tidak (precision rendah).

Karena data tidak seimbang, akurasi saja tidak cukup. Evaluasi menitikberatkan pada **F1-score, recall, dan ROC-AUC**.

**Keterbatasan dan pengembangan**
- Hasil tuning `C` sama dengan nilai bawaan, sehingga tuning tidak memperbaiki model. Tuning belum dilakukan untuk model lain.
- Perbandingan model memakai threshold tetap 0,5, yang merugikan Random Forest (ROC-AUC CV-nya lebih tinggi). Pencarian threshold per model dapat dicoba pada pengembangan berikutnya.

## 📈 Forecasting Order

Folder `Forecasting/` berisi data historis order bulanan, model ARIMA, dan hasil forecast 3 bulan ke depan (Desember 2025 sampai Februari 2026) beserta interval kepercayaannya. Hasilnya ditampilkan pada tab **Forecast Order** di aplikasi.

## 🖥️ Fitur Aplikasi

| Tab | Isi |
|---|---|
| 📊 Ringkasan & EDA | KPI, konversi per bulan dan tipe pengunjung, boxplot, korelasi (data training) |
| 📈 Forecast Order | Tren order historis, forecast 3 bulan, completion rate |
| 🤖 Performa Model | Metrik test, perbandingan model, confusion matrix, pemilihan threshold, koefisien model |
| 🎯 Prediksi | Form input sesi pengunjung, skor peluang membeli, dan rekomendasi aksi |

## 📁 Struktur Repository

```
ConvertIQ/
├── app.py                          # aplikasi Streamlit
├── train_model.py                  # mereproduksi model final notebook dan menyimpannya
├── feature_engineering.py          # fungsi pembuatan fitur baru
├── requirements.txt                # daftar library
├── convertiq_model.joblib          # model final + threshold + daftar fitur (dipakai app.py)
├── model_meta.json                 # hasil evaluasi untuk tab Performa Model
├── convertiq_dashboard_data.csv    # dataset bersih (12.205 sesi)
├── monthly_orders.csv              # data order bulanan
├── historical_orders.csv           # data historis order
├── forecast_results.csv            # hasil forecast
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
| _(Raditya Aji Respati Juminar)_ |
| _(Najwa Aulia Syahrani)_ |
| _(Alya Darmawanti Armunanto)_ |
| _(Nagita Nur Febrina)_ |

---

_Dibuat untuk Learning Path Data ISCOM 2026._
