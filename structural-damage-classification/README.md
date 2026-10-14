# Structural Damage Classification (Kaggle)

Tugas 1 - Supervised Machine Learning. Memprediksi tingkat kerusakan bangunan (`damage_grade` 1-5) pasca gempa Gorkha, Nepal (25 April 2015) dari karakteristik struktural bangunan.

- **Kompetisi:** https://www.kaggle.com/competitions/structural-damage-classification
- **Metrik:** Macro F1-Score
- **Batasan:** tanpa Neural Network; preprocessing, training, dan tuning memakai `Pipeline`.

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
│   ├── visualize.py            # 12 fungsi visualisasi EDA dan evaluasi -> reports/figures/
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

- **Distribusi target dan dampaknya:** kelas tidak seimbang; grade 5 paling banyak (36,2%) dan grade 1-2 paling sedikit (sekitar 10-11%). Karena Macro F1 memberi bobot sama pada tiap kelas, model memakai `class_weight="balanced"` dan bobot kelas tambahan pada probabilitas. Distribusi prediksi test (`1: 10,1%, 2: 13,3%, 3: 19,7%, 4: 24,3%, 5: 32,6%`) mirip distribusi data latih.
- **Model terbaik:** Logistic Regression unggul atas HistGradientBoosting dan Random Forest pada subset (0.5006 vs 0.4851 vs 0.4824). Tuning hampir tidak mengubah skor (0.5006 -> 0.5007), sedangkan bobot kelas menaikkan Macro F1 out-of-fold dari 0.5158 menjadi 0.5224 (+0.0066). **TODO:** tambahkan dugaan penyebab model linear unggul (mis. cara encoding fitur wilayah) setelah dicek.
- **Kesalahan prediksi:** F1 tertinggi ada pada grade 5 (0.713) dan grade 1 (0.620); terendah pada grade 2 (0.374) dan grade 3 (0.411). Kelas tengah paling sulit dibedakan. **TODO:** jelaskan pola tertukar dari `reports/figures/confusion_matrix.png` dan hubungkan dengan sifat ordinal `damage_grade`.
- **Fitur penting:** **TODO** (lihat `reports/figures/feature_importance.png`; hubungkan dengan logika struktural, mis. umur bangunan, jenis fondasi, material superstructure, jumlah lantai, lokasi).
- **Keterbatasan dan perbaikan:** Macro F1 CV (0.5158-0.5224) dekat dengan skor public (0.52) dan simpangan antar fold kecil (0.0019), jadi tidak ada tanda overfitting besar. Bobot kelas dioptimasi pada prediksi out-of-fold sehingga tetap berpotensi sedikit terlalu menyesuaikan data latih. **TODO:** tulis ide perbaikan yang kamu coba atau rencanakan.

## Catatan

- Jangan commit dataset besar; `data/*.csv` sudah masuk `.gitignore`.
- Setiap function/class di `scripts/` memiliki docstring.
- Random seed tetap (`42`) untuk reprodusibilitas.
