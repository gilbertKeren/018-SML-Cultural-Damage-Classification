# Structural Damage Classification (Kaggle)

Tugas - Supervised Machine Learning. Memprediksi tingkat kerusakan bangunan (`damage_grade` 1-5) pasca gempa Gorkha, Nepal (25 April 2015) dari karakteristik struktural bangunan.

## Struktur Repository

```
structural-damage-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                     # train.csv, test.csv, sample_submission.csv (tidak di-commit)
├── notebooks/
│   ├── 01_eda.ipynb          # eksplorasi data
│   ├── 02_modeling.ipynb     # perbandingan model, tuning, evaluasi, feature importance
│   └── 03_submission.ipynb   # membuat dan mengirim submission
├── scripts/
│   ├── __init__.py
│   ├── feature_engineering.py  # konstanta kolom, fitur turunan, encoder superstructure
│   ├── preprocess.py           # load data, ColumnTransformer
│   ├── train.py                # pipeline, CV, tuning, bobot kelas, simpan model
│   ├── visualize.py            # fungsi visualisasi EDA dan evaluasi
│   └── predict.py              # prediksi test, validasi format, submit via CLI
├── models/                   # pipeline_model_<nama>.pkl, class_weights_<nama>.json
├── submissions/              # submission.csv
└── reports/
    └── figures/              # gambar hasil analisis
```

## Cara Menggunakan

### 1. Instalasi

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Unduh data

Dataset tidak di-commit (ukuran > 100 MB). Pastikan akun Kaggle sudah terverifikasi telepon dan peraturan kompetisi sudah di-accept lewat halaman web (**Join Competition -> I Understand and Accept**).

```bash
# letakkan kaggle.json di ~/.kaggle/ (Kaggle > Settings > API > Create New Token)
kaggle competitions download -c structural-damage-classification -p data/
unzip data/structural-damage-classification.zip -d data/
```

Atau klik **Download All** di tab *Data* lalu ekstrak ke folder `data/`.

### 3. Jalankan analisis

Urutan notebook: `01_eda.ipynb` -> `02_modeling.ipynb` -> `03_submission.ipynb` (jalankan dari folder `notebooks/`). Gambar hasil analisis otomatis tersimpan di `reports/figures/`, model di `models/`, dan submission di `submissions/`.

Notebook mendeteksi sendiri apakah dijalankan di Kaggle (data dibaca dari `/kaggle/input`, output ke `/kaggle/working`) atau lokal (data dari `data/`). Di Kaggle, folder `scripts/` harus tersedia di `/kaggle/working/` (mis. `git clone` repo ini atau upload `scripts/` sebagai Dataset lalu disalin ke sana).

### 4. Alternatif lewat command line

Jalankan dari root repo:

```bash
# eksperimen cepat pada subset
python -m scripts.train --model reglog --sample 100000 --cv 3

# training penuh: CV 5-fold + optimasi bobot kelas, model disimpan ke models/
python -m scripts.train --model reglog --cv 5 --opt-weights

# dengan tuning hyperparameter
python -m scripts.train --model reglog --tune --n-iter 12 --opt-weights

# buat submissions/submission.csv (otomatis memakai model terakhir yang disimpan
# dan bobot kelasnya; pilih model lain dengan --model rf / --model hgb)
python -m scripts.predict

# buat sekaligus kirim ke Kaggle
python -m scripts.predict --submit --message "reglog tuned + class weights"
```

Model yang tersedia: `reglog` (Logistic Regression multinomial), `rf` (Random Forest), `hgb` (HistGradientBoosting).

### 5. Submit manual

Upload `submissions/submission.csv` lewat tombol **Submit Prediction**, atau:

```bash
kaggle competitions submit -c structural-damage-classification -f submissions/submission.csv -m "deskripsi"
```

Maksimal 5 submission per hari; pilih maksimal 2 submission final sebelum kompetisi berakhir.

## Workflow Analisis

1. **EDA** - tipe data, nilai kosong, distribusi target, outlier numerik (mis. kode khusus pada `age_building`), hubungan kategori dengan kerusakan, format `superstructure`, kardinalitas fitur geografis.
2. **Preprocessing** (`scripts/preprocess.py`, `ColumnTransformer` di dalam `Pipeline`)
   - `building_id` dibuang dari fitur (hanya dipakai pada submission).
   - Numerik: imputasi median (+ `StandardScaler` untuk model linear).
   - Kategorikal dan id wilayah (`district_id`, `vdcmun_id`, `ward_id`) diperlakukan sebagai kategori, bukan angka. Model linear: One-Hot (kategori jarang digabung); model pohon: Ordinal Encoding.
   - `superstructure`: multi-hot per material (dipisah oleh `,;|/+&`; sesuaikan `split_materials` jika hasil EDA berbeda).
3. **Feature engineering** (`scripts/feature_engineering.py`): `area_per_floor`, `height_per_floor`, `area_to_height`, `age_log`.
4. **Modeling**: Logistic Regression (baseline), Random Forest, HistGradientBoosting dengan `class_weight` seimbang karena target tidak seimbang dan metriknya Macro F1.
5. **Validasi**: Stratified K-Fold dengan Macro F1; tuning `RandomizedSearchCV` di dalam pipeline sehingga tidak ada data leakage.
6. **Post-processing**: bobot per kelas pada probabilitas dioptimasi pada prediksi out-of-fold untuk memaksimalkan Macro F1.
7. **Submission**: prediksi test, validasi format terhadap `sample_submission.csv`, upload ke Kaggle.

## Laporan Interpretasi

> Angka di bawah berasal dari run di Kaggle (533.465 baris latih, seed 42). Bagian bertanda **TODO** perlu dilengkapi dari gambar di `reports/figures/`.

### Hasil validasi

| Model | Macro F1 CV (mean +/- std) | Catatan |
|-------|----------------------------|---------|
| Logistic Regression | 0.5006 +/- 0.0013 | subset 100.000 baris, 3-fold; terbaik |
| HistGradientBoosting | 0.4851 +/- 0.0033 | subset 100.000 baris, 3-fold |
| Random Forest | 0.4824 +/- 0.0026 | subset 100.000 baris, 3-fold |
| Logistic Regression, tuning (C = 2.54) | 0.5007 | `RandomizedSearchCV`, 12 kombinasi, 3-fold |
| Logistic Regression pada seluruh data | 0.5158 +/- 0.0019 | CV 5-fold |
| + bobot kelas `[0.5, 0.8, 1.0, 1.1, 1.2]` | 0.5224 | Macro F1 out-of-fold; model final |

Skor leaderboard: Public 0.52 | Private: diisi setelah kompetisi berakhir.

Metrik per kelas pada prediksi out-of-fold (dengan bobot kelas):

| damage_grade | Precision | Recall | F1 | Jumlah | Proporsi |
|--------------|-----------|--------|----|--------|----------|
| 1 | 0.631 | 0.610 | 0.620 | 55.170 | 10,3% |
| 2 | 0.347 | 0.406 | 0.374 | 61.080 | 11,4% |
| 3 | 0.394 | 0.431 | 0.411 | 95.488 | 17,9% |
| 4 | 0.490 | 0.498 | 0.494 | 128.691 | 24,1% |
| 5 | 0.753 | 0.677 | 0.713 | 193.036 | 36,2% |

### Interpretasi

### Interpretasi
---

## Distribusi Target dan Dampaknya
- Kelas tidak seimbang:
  - **Grade 5** paling banyak: **36,2%**
  - **Grade 4**: 24,1%
  - **Grade 3**: 17,9%
  - **Grade 2**: 11,4%
  - **Grade 1–2** paling sedikit: sekitar **10–11%**
- Karena **Macro F1** memberi bobot sama pada tiap kelas, model memakai `class_weight="balanced"` dan bobot kelas tambahan pada probabilitas.
- Distribusi prediksi test:
  - Grade 1: 10,1%
  - Grade 2: 13,3%
  - Grade 3: 19,7%
  - Grade 4: 24,3%
  - Grade 5: 32,6%
- Distribusi prediksi test mirip dengan distribusi data latih.
- **Macro F1** lebih tepat karena memberi bobot sama ke setiap kelas.
- Nilai **Macro F1 akhir 0.5224** menunjukkan performa masih moderat.

---

## Model Terbaik
- **Logistic Regression** unggul atas HistGradientBoosting dan Random Forest pada subset:
  - Reglog: **0.5006**
  - HGB: **0.4851**
  - RF: **0.4824**
- Tuning hampir tidak mengubah skor: **0.5006 → 0.5007**
- Bobot kelas menaikkan Macro F1 out-of-fold dari **0.5158 → 0.5224** (**+0.0066**).
- Tuning hyperparameter `C` dan penambahan bobot kelas meningkatkan performa serta menjaga stabilitas model.
- Reglog juga relatif lebih interpretatif untuk melihat arah pengaruh fitur.

### Parameter Terbaik
- Model: `reglog`
- `C`: **2.5378155082656657**
- `cv_folds`: **5**
- `cv_macro_f1_mean`: **0.5157685913019705**
- `cv_macro_f1_std`: **0.0018824964668821419**
- `macro_f1_oof_with_class_weights`: **0.5224257064146529**
- `class_weights`: **[0.5, 0.8, 1.0, 1.1, 1.2]**

### Perbandingan Model
| Model | Macro F1 Mean | Macro F1 Std |
|---|---:|---:|
| reglog | 0.500636 | 0.001305 |
| hgb | 0.485054 | 0.003285 |
| rf | 0.482402 | 0.002594 |

---

## Evaluasi dan Kesalahan Prediksi
### Classification Report
| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 1 | 0.631 | 0.610 | 0.620 | 55.170 |
| 2 | 0.347 | 0.406 | 0.374 | 61.080 |
| 3 | 0.394 | 0.431 | 0.411 | 95.488 |
| 4 | 0.490 | 0.498 | 0.494 | 128.691 |
| 5 | 0.753 | 0.677 | 0.713 | 193.036 |
| **Accuracy** | | | **0.551** | **533.465** |
| **Macro Avg** | 0.523 | 0.524 | 0.522 | 533.465 |
| **Weighted Avg** | 0.566 | 0.551 | 0.558 | 533.465 |

### Confusion Matrix (Recall per Kelas)
| Aktual \ Prediksi | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| **1** | 0.61 | 0.26 | 0.07 | 0.03 | 0.02 |
| **2** | 0.17 | 0.41 | 0.26 | 0.11 | 0.05 |
| **3** | 0.05 | 0.18 | 0.43 | 0.23 | 0.10 |
| **4** | 0.02 | 0.07 | 0.20 | 0.50 | 0.22 |
| **5** | 0.01 | 0.04 | 0.09 | 0.19 | 0.68 |

### Interpretasi Kesalahan
- F1 tertinggi ada pada **grade 5 (0.713)** dan **grade 1 (0.620)**.
- F1 terendah ada pada **grade 2 (0.374)** dan **grade 3 (0.411)**.
- **Kelas tengah paling sulit dibedakan.**
- Model cenderung "ragu" pada kelas-kelas yang berbatasan langsung.
- Kesalahan terbesar selalu terjadi pada **kelas tetangga** (1↔2, 2↔3, 3↔4, 4↔5).
- Ini adalah ciri khas **klasifikasi ordinal (bertingkat)**.
- Model tahu bahwa grade 2 lebih buruk dari grade 1, tetapi kesulitan menemukan garis pemisah yang tepat antara "rusak ringan" dan "rusak sedang".
- Secara struktural, perbedaan antara grade 2 dan 3, atau 3 dan 4, mungkin tidak terlalu ekstrem secara fisik.
- Perbedaannya bisa jadi bersifat subjektif saat survei atau dipengaruhi kombinasi faktor yang kompleks.

---

## Fitur Penting
Fitur geografis mendominasi:
- `district_id` (0.107532)
- `ward_id` (0.103773)
- `vdcmun_id` (0.068662)

Secara struktural, ini masuk akal sebagai proxy kondisi wilayah:
- Praktik konstruksi lokal
- Kode bangunan
- Tingkat bahaya gempa
- Jenis tanah
- Sosial-ekonomi

Namun, dominasi ini juga **berisiko overfitting spasial**.

### Fitur Struktural
- `superstructure` adalah fitur struktural paling masuk akal.
- Dari grafik material:
  - `rc_engineered` dominan grade 1, hampir tanpa grade 5.
  - `rc_non_engineered` juga relatif aman.
  - `stone_flag`, `mud_mortar_stone`, `adobe_mud`, `mud_mortar_brick` didominasi grade 4–5.
- Urutan dari paling aman ke paling rusak:
  **rc_engineered > rc_non_engineered > cement_mortar_brick > bamboo > timber > cement_mortar_stone > mud_mortar_brick > adobe_mud > mud_mortar_stone > stone_flag**

### Korelasi Fitur Numerik dengan `damage_grade`
| Fitur | Korelasi Spearman |
|---|---:|
| `age_building` | 0.241 |
| `count_floors_pre_eq` | 0.198 |
| `height_ft_pre_eq` | 0.095 |
| `plinth_area_sq_ft` | -0.115 |

- `age_building`: semakin tua bangunan, cenderung semakin rusak. Masuk akal karena degradasi material dan standar bangunan lama.
- `count_floors_pre_eq`: bangunan lebih banyak lantai cenderung lebih rentan.
- `height_ft_pre_eq`: bangunan lebih tinggi sedikit lebih rentan.
- `plinth_area_sq_ft`: luas plinth lebih besar justru sedikit berkorelasi dengan grade lebih rendah. Ini bisa jadi proxy bangunan lebih besar/engineered, bukan sebab langsung.

---

## Keterbatasan
- Macro F1 masih sekitar **0.52** dan accuracy **0.551**; model belum kuat untuk membedakan kelas tengah.
- **Kelas 2 dan 3 paling sulit** dengan precision dan recall rendah.
- Dominasi fitur ID wilayah membuat model berisiko tidak general ke wilayah baru, meskipun di test tidak ada ID baru.
- **Kardinalitas tinggi**:
  - `ward_id`: 945 kategori
  - `vdcmun_id`: 110 kategori
  - `district_id`: 11 kategori
  - One-hot encoding akan sparse dan rawan overfit.
- **Kode khusus dan outlier**:
  - `age_building` memiliki nilai **999**
  - `height_ft_pre_eq` memiliki nilai **99**
  - `plinth_area_sq_ft` maksimum **5000**
  - `count_floors_pre_eq` maksimum **9**
- **Multikolinearitas**:
  - `count_floors_pre_eq` vs `height_ft_pre_eq` = **0.71**
  - `area_per_floor` vs `area_to_height` = **0.88**
  - `age_building` vs `age_log` = **1.00**

---

## Rekomendasi Perbaikan
- **Ordinal modeling**: karena `damage_grade` bersifat ordinal 1–5, gunakan ordinal logistic/regression atau model yang menghormati urutan kelas.
- **Encoding geografis yang lebih baik**: target encoding/frequency encoding dengan smoothing dan cross-fitting untuk `district_id`, `vdcmun_id`, `ward_id`.
- **Spatial cross-validation**: group CV berdasarkan wilayah untuk mengukur generalisasi ke wilayah baru.
- **Model non-linear**: coba LightGBM/XGBoost/CatBoost dengan tuning lebih serius dan encoding kategorikal yang tepat.
- **Feature engineering**: buat interaksi seperti:
  - `superstructure × age_building`
  - `height_ft_pre_eq × plinth_area_sq_ft`
  - `foundation_type × ground_floor_type`
  - rasio tinggi per luas
  - umur per material
- **Penanganan imbalance**: threshold tuning per kelas, focal loss, resampling, atau bobot kelas yang lebih agresif.
- **Handling kode khusus/outlier**: ubah 999 dan 99 menjadi missing, tambahkan indikator missing, lakukan imputasi median/modus, dan winsorizing/log transform.
- **Evaluasi lebih lengkap**: tampilkan confusion matrix, per-class F1, PR-AUC, ROC-AUC, dan kalibrasi probabilitas.
- **Cost-sensitive learning**: jika tujuan praktis adalah mencegah prediksi kerusakan rendah yang fatal, beri penalti lebih besar untuk kesalahan prediksi grade rendah.
- **Ensemble**: gabungkan reglog dengan model tree-based untuk menangkap pola linear dan non-linear sekaligus.

---

## Catatan

- Jangan commit dataset besar; `data/*.csv` sudah masuk `.gitignore`.
- Setiap function/class di `scripts/` memiliki docstring.
- Random seed tetap (`42`) untuk reprodusibilitas.
