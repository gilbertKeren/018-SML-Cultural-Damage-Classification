"""Rekayasa fitur: konstanta kolom, fitur turunan, dan encoder kolom ``superstructure``."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ID_COL = "building_id"
TARGET_COL = "damage_grade"

GEO_COLS = ["district_id", "vdcmun_id", "ward_id"]
CATEGORICAL_COLS = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
]
SUPERSTRUCTURE_COL = "superstructure"

BASE_NUMERIC_COLS = [
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "height_ft_pre_eq",
]
ENGINEERED_NUMERIC_COLS = [
    "area_per_floor",
    "height_per_floor",
    "area_to_height",
    "age_log",
]
NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

# Pemisah material pada kolom superstructure. Sesuaikan jika hasil EDA
# menunjukkan format yang berbeda (mis. dipisah spasi atau tanda lain).
_SEPARATORS = re.compile(r"[,;|/+&]")


def split_materials(text: str) -> list[str]:
    """Pecah satu string ``superstructure`` menjadi daftar material.

    Parameters
    ----------
    text : str
        Nilai mentah kolom ``superstructure``, mis. ``"Mud mortar-Stone/Brick"``.

    Returns
    -------
    list of str
        Token material (huruf kecil, tanpa spasi tepi, tanpa token kosong).
    """
    parts = [p.strip().lower() for p in _SEPARATORS.split(text)]
    return [p for p in parts if p]


def build_superstructure_encoder() -> CountVectorizer:
    """Buat encoder multi-hot untuk kolom ``superstructure``.

    Jika satu bangunan memiliki beberapa material, setiap material menjadi
    satu kolom biner. Jika nilainya hanya satu token, hasilnya setara one-hot.

    Returns
    -------
    sklearn.feature_extraction.text.CountVectorizer
        Encoder biner dengan tokenizer ``split_materials``.
    """
    return CountVectorizer(
        tokenizer=split_materials,
        token_pattern=None,
        lowercase=False,
        binary=True,
        min_df=5,
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan dan rapikan tipe data kolom.

    Langkah yang dilakukan:

    * kolom numerik dipaksa ke angka (nilai tak valid menjadi NaN),
    * ``area_per_floor``, ``height_per_floor``, ``area_to_height`` dan
      ``age_log`` dihitung (pembagian dengan nol menjadi NaN),
    * kolom geografis dan kategorikal diubah ke string agar encoder
      memperlakukannya sebagai kategori, bukan angka,
    * nilai kosong ``superstructure`` diisi ``"unknown"``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data mentah (train atau test) tanpa ``damage_grade``.

    Returns
    -------
    pandas.DataFrame
        Salinan ``df`` dengan kolom tambahan dan tipe yang sudah dirapikan.
    """
    out = df.copy()
    for col in BASE_NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    floors = out["count_floors_pre_eq"].replace(0, np.nan)
    height = out["height_ft_pre_eq"].replace(0, np.nan)
    out["area_per_floor"] = out["plinth_area_sq_ft"] / floors
    out["height_per_floor"] = out["height_ft_pre_eq"] / floors
    out["area_to_height"] = out["plinth_area_sq_ft"] / height
    out["age_log"] = np.log1p(out["age_building"].clip(lower=0))
    out[ENGINEERED_NUMERIC_COLS] = out[ENGINEERED_NUMERIC_COLS].replace(
        [np.inf, -np.inf], np.nan
    )

    for col in GEO_COLS + CATEGORICAL_COLS:
        out[col] = out[col].astype(str)
    out[SUPERSTRUCTURE_COL] = out[SUPERSTRUCTURE_COL].fillna("unknown").astype(str)
    return out
