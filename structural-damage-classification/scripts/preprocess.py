"""Preprocessing: memuat data dan membangun ``ColumnTransformer``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from scripts.feature_engineering import (
    CATEGORICAL_COLS,
    GEO_COLS,
    ID_COL,
    NUMERIC_COLS,
    SUPERSTRUCTURE_COL,
    TARGET_COL,
    build_superstructure_encoder,
)


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Muat ``train.csv``, ``test.csv`` dan ``sample_submission.csv``.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Folder berisi file kompetisi.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, test, sample_submission)``.
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


def split_xy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pisahkan fitur dan target dari data latih.

    ``building_id`` dibuang karena hanya pengenal. Jika target berbentuk teks
    (mis. ``"Grade 3"``), angkanya diekstrak menjadi integer 1-5.

    Parameters
    ----------
    train : pandas.DataFrame
        Data latih lengkap dengan kolom ``damage_grade``.

    Returns
    -------
    tuple
        ``(X, y)`` dengan indeks yang sudah di-reset.
    """
    X = train.drop(columns=[ID_COL, TARGET_COL]).reset_index(drop=True)
    y = train[TARGET_COL].reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y.astype(str).str.extract(r"(\d+)")[0])
    return X, y.astype(int)


def build_preprocessor(kind: str = "linear") -> ColumnTransformer:
    """Bangun ``ColumnTransformer`` sesuai jenis model.

    Parameters
    ----------
    kind : {"linear", "tree"}
        ``"linear"``: numerik di-scale, kategori di-one-hot (kategori jarang
        digabung). ``"tree"``: numerik hanya di-impute, kategori di-ordinal
        encode, output selalu dense untuk model berbasis pohon.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformer yang mengharapkan kolom hasil
        :func:`scripts.feature_engineering.add_engineered_features`.
    """
    if kind not in {"linear", "tree"}:
        raise ValueError("kind harus 'linear' atau 'tree'")

    if kind == "linear":
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=20
        )
        geo = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)
        sparse_threshold = 0.3
    else:
        numeric = SimpleImputer(strategy="median")
        categorical = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        geo = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        sparse_threshold = 0.0

    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
            ("geo", geo, GEO_COLS),
            ("sup", build_superstructure_encoder(), SUPERSTRUCTURE_COL),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )
