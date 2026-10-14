"""Prediksi data test, validasi format, dan (opsional) submit ke Kaggle.

Contoh pemakaian (dari root repo)::

    python -m scripts.predict                      # memakai model terakhir di models/
    python -m scripts.predict --model reglog --submit --message "reglog + class weights"
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.feature_engineering import ID_COL, TARGET_COL

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "structural-damage-classification"


def make_submission(
    model_path: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    weights_path: str | Path | None = None,
) -> pd.DataFrame:
    """Buat ``submission.csv`` dari pipeline tersimpan dan validasi formatnya.

    Parameters
    ----------
    model_path : str or pathlib.Path
        File ``.pkl`` pipeline hasil ``scripts.train``.
    data_dir : str or pathlib.Path
        Folder berisi ``test.csv`` dan ``sample_submission.csv``.
    output_path : str or pathlib.Path
        Lokasi file CSV keluaran.
    weights_path : str or pathlib.Path, optional
        JSON bobot kelas. Jika ada, probabilitas dikali bobot sebelum ``argmax``.

    Returns
    -------
    pandas.DataFrame
        Isi submission (kolom ``building_id`` dan ``damage_grade``).
    """
    data_dir = Path(data_dir)
    pipeline = joblib.load(model_path)
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    proba = pipeline.predict_proba(test.drop(columns=[ID_COL]))
    if weights_path is not None and Path(weights_path).exists():
        proba = proba * np.array(json.loads(Path(weights_path).read_text()))
        print(f"Bobot kelas dipakai dari {weights_path}")
    pred = pipeline.classes_[proba.argmax(axis=1)].astype(int)

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: pred})
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")  # ikuti urutan sample

    assert list(sub.columns) == list(sample.columns), "kolom tidak sama dengan sample"
    assert len(sub) == len(sample), "jumlah baris tidak sama dengan sample"
    assert sub[TARGET_COL].notna().all(), "ada building_id sample tanpa prediksi"
    assert sub[ID_COL].is_unique, "building_id duplikat"
    sub[TARGET_COL] = sub[TARGET_COL].astype(int)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"Submission disimpan: {output_path} ({len(sub)} baris)")
    print("Distribusi prediksi:", sub[TARGET_COL].value_counts(normalize=True).sort_index().round(3).to_dict())
    return sub


def submit_to_kaggle(competition: str, file_path: str | Path, message: str) -> None:
    """Kirim file submission lewat Kaggle CLI.

    Syarat: ``kaggle.json`` sudah terpasang dan peraturan kompetisi sudah
    di-accept lewat halaman web.

    Parameters
    ----------
    competition : str
        Slug kompetisi.
    file_path : str or pathlib.Path
        File CSV submission.
    message : str
        Deskripsi submission.
    """
    cmd = ["kaggle", "competitions", "submit", "-c", competition,
           "-f", str(file_path), "-m", message]
    print("Menjalankan:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Titik masuk CLI untuk membuat (dan opsional mengirim) submission."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None,
                        help="nama model (reglog/rf/hgb); default: model yang terakhir disimpan")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "submissions" / "submission.csv"))
    parser.add_argument("--no-weights", action="store_true", help="abaikan bobot kelas")
    parser.add_argument("--submit", action="store_true", help="kirim ke Kaggle via CLI")
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--message", default="submission")
    args = parser.parse_args()

    if args.model is None:
        saved = sorted((ROOT / "models").glob("pipeline_model_*.pkl"), key=lambda p: p.stat().st_mtime)
        if not saved:
            raise SystemExit("Belum ada model di models/. Jalankan scripts.train dulu.")
        args.model = saved[-1].stem.replace("pipeline_model_", "")
        print("Memakai model terbaru:", args.model)
    model_path = ROOT / "models" / f"pipeline_model_{args.model}.pkl"
    weights_path = None if args.no_weights else ROOT / "models" / f"class_weights_{args.model}.json"
    make_submission(model_path, args.data_dir, args.output, weights_path)
    if args.submit:
        submit_to_kaggle(args.competition, args.output, args.message)


if __name__ == "__main__":
    main()
