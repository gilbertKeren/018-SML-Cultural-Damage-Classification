"""Pelatihan model: pipeline, cross-validation, tuning, optimasi bobot kelas, penyimpanan.

Contoh pemakaian (dari root repo)::

    python -m scripts.train --model hgb --cv 5 --opt-weights
    python -m scripts.train --model rf --tune --n-iter 15
    python -m scripts.train --model reglog --sample 100000   # eksperimen cepat
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from scripts.feature_engineering import add_engineered_features
from scripts.preprocess import build_preprocessor, load_data, split_xy

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_NAMES = ("reglog", "rf", "hgb")


def build_estimator(model_name: str, random_state: int = 42):
    """Buat estimator (bukan Neural Network) sesuai nama.

    Parameters
    ----------
    model_name : {"reglog", "rf", "hgb"}
        ``reglog`` = Logistic Regression multinomial, ``rf`` = Random Forest,
        ``hgb`` = HistGradientBoosting. Semuanya memakai pembobotan kelas
        ``balanced`` karena metrik evaluasi adalah Macro F1.
    random_state : int
        Seed reprodusibilitas.

    Returns
    -------
    sklearn estimator
    """
    if model_name == "reglog":
        return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=400,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )
    raise ValueError(f"model_name harus salah satu dari {MODEL_NAMES}")


def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Bangun pipeline lengkap: fitur turunan -> preprocessing -> model.

    Semua langkah berada di dalam satu ``Pipeline`` sehingga tidak ada
    kebocoran data saat cross-validation maupun tuning.

    Parameters
    ----------
    model_name : {"reglog", "rf", "hgb"}
    random_state : int

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    kind = "linear" if model_name == "reglog" else "tree"
    return Pipeline(
        [
            ("features", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocess", build_preprocessor(kind)),
            ("model", build_estimator(model_name, random_state)),
        ]
    )


def get_param_distributions(model_name: str) -> dict:
    """Ruang pencarian hyperparameter untuk ``RandomizedSearchCV``.

    Parameters
    ----------
    model_name : {"reglog", "rf", "hgb"}

    Returns
    -------
    dict
        Kunci berformat ``model__<param>`` sesuai nama step di pipeline.
    """
    if model_name == "reglog":
        return {"model__C": loguniform(1e-2, 1e2)}
    if model_name == "rf":
        return {
            "model__n_estimators": [200, 300, 500],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.3],
            "model__max_depth": [None, 20, 30],
        }
    if model_name == "hgb":
        return {
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__max_leaf_nodes": [31, 63, 127],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": loguniform(1e-3, 10),
            "model__max_iter": [300, 500],
        }
    raise ValueError(f"model_name harus salah satu dari {MODEL_NAMES}")


def subsample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = 42):
    """Ambil subset stratified untuk eksperimen cepat.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    n : int
        Jumlah baris yang diinginkan. Jika ``n >= len(X)`` data dikembalikan utuh.
    seed : int

    Returns
    -------
    tuple
        ``(X_sub, y_sub)`` dengan indeks yang sudah di-reset.
    """
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def tune_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_iter: int = 15,
    cv: int = 3,
    seed: int = 42,
) -> tuple[Pipeline, dict, float]:
    """Tuning hyperparameter dengan ``RandomizedSearchCV`` (skor Macro F1).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    X, y : data latih
    model_name : {"reglog", "rf", "hgb"}
    n_iter : int
        Jumlah kombinasi yang dicoba.
    cv : int
        Jumlah fold stratified.
    seed : int

    Returns
    -------
    tuple
        ``(pipeline_dengan_param_terbaik, best_params, best_cv_score)``.
        Pipeline belum di-fit pada seluruh data.
    """
    search = RandomizedSearchCV(
        pipeline,
        get_param_distributions(model_name),
        n_iter=n_iter,
        scoring="f1_macro",
        cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
        random_state=seed,
        refit=False,
        verbose=1,
    )
    search.fit(X, y)
    tuned = clone(pipeline).set_params(**search.best_params_)
    return tuned, search.best_params_, float(search.best_score_)


def macro_f1_fast(y_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> float:
    """Hitung Macro F1 cepat dari indeks kelas (setara ``f1_score(average='macro')``).

    Parameters
    ----------
    y_idx, pred_idx : numpy.ndarray
        Label aktual dan prediksi dalam bentuk indeks 0..n_classes-1.
    n_classes : int

    Returns
    -------
    float
    """
    cm = np.bincount(y_idx * n_classes + pred_idx, minlength=n_classes**2)
    cm = cm.reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def cross_validate_macro_f1(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Stratified K-Fold CV dengan Macro F1 dan probabilitas out-of-fold.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline yang belum di-fit (di-clone untuk setiap fold).
    X : pandas.DataFrame
    y : pandas.Series
    n_splits : int
    seed : int
    verbose : bool
        Cetak skor tiap fold.

    Returns
    -------
    dict
        ``fold_scores``, ``mean``, ``std``, ``oof_proba`` (n x kelas) dan ``classes``.
    """
    y_arr = np.asarray(y)
    classes = np.sort(np.unique(y_arr))
    oof = np.zeros((len(y_arr), len(classes)))
    scores = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y_arr), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr], y_arr[tr])
        proba = model.predict_proba(X.iloc[va])
        oof[va] = proba
        pred = model.classes_[proba.argmax(axis=1)]
        score = f1_score(y_arr[va], pred, average="macro")
        scores.append(score)
        if verbose:
            print(f"  fold {fold}/{n_splits}: Macro F1 = {score:.4f}")
    return {
        "fold_scores": [float(s) for s in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "oof_proba": oof,
        "classes": classes,
    }


def optimize_class_weights(
    proba: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    n_rounds: int = 3,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Cari bobot per kelas pada probabilitas agar Macro F1 maksimum.

    Coordinate ascent: tiap kelas dicoba beberapa pengali secara bergantian.
    Gunakan probabilitas *out-of-fold* agar bobot tidak dioptimasi pada data
    yang sudah dilihat model.

    Parameters
    ----------
    proba : numpy.ndarray
        Probabilitas prediksi, bentuk ``(n, n_classes)``.
    y_true : numpy.ndarray
        Label aktual (nilai asli, mis. 1-5).
    classes : numpy.ndarray
        Urutan kelas pada kolom ``proba``.
    n_rounds : int
        Jumlah putaran coordinate ascent.
    grid : numpy.ndarray, optional
        Kandidat pengali. Default ``linspace(0.5, 3.0, 26)``.

    Returns
    -------
    tuple
        ``(weights, best_macro_f1)``.
    """
    if grid is None:
        grid = np.linspace(0.5, 3.0, 26)
    n_classes = len(classes)
    y_idx = np.searchsorted(classes, np.asarray(y_true))
    weights = np.ones(n_classes)
    best = macro_f1_fast(y_idx, (proba * weights).argmax(axis=1), n_classes)
    for _ in range(n_rounds):
        for k in range(n_classes):
            for g in grid:
                cand = weights.copy()
                cand[k] = g
                score = macro_f1_fast(y_idx, (proba * cand).argmax(axis=1), n_classes)
                if score > best:
                    best, weights = score, cand
    return weights, float(best)


def save_pipeline(
    pipeline: Pipeline, model_name: str, weights: np.ndarray | None = None
) -> Path:
    """Simpan pipeline (``.pkl``) dan bobot kelas (``.json``, opsional) ke ``models/``.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline yang sudah di-fit.
    model_name : str
    weights : numpy.ndarray, optional

    Returns
    -------
    pathlib.Path
        Lokasi file model.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"pipeline_model_{model_name}.pkl"
    joblib.dump(pipeline, model_path, compress=3)
    weights_path = MODEL_DIR / f"class_weights_{model_name}.json"
    if weights is not None:
        weights_path.write_text(json.dumps([float(w) for w in weights]))
    elif weights_path.exists():
        weights_path.unlink()  # hindari bobot lama yang tidak cocok dengan model baru
    return model_path


def main() -> None:
    """Titik masuk CLI: CV, (opsional) tuning dan bobot kelas, fit akhir, simpan model."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=MODEL_NAMES, default="hgb")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--cv", type=int, default=5, help="jumlah fold CV")
    parser.add_argument("--tune", action="store_true", help="aktifkan RandomizedSearchCV")
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--opt-weights", action="store_true", help="optimasi bobot kelas")
    parser.add_argument("--sample", type=int, default=0, help="subset stratified (0 = semua)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()
    train, _, _ = load_data(args.data_dir)
    X, y = split_xy(train)
    if args.sample:
        X, y = subsample(X, y, args.sample, args.seed)
    print(f"Data latih: {X.shape[0]} baris, {X.shape[1]} fitur | model: {args.model}")
    print("Distribusi kelas:", y.value_counts(normalize=True).sort_index().round(3).to_dict())

    pipeline = build_pipeline(args.model, args.seed)
    best_params: dict = {}
    if args.tune:
        print(f"Tuning ({args.n_iter} kombinasi)...")
        pipeline, best_params, tune_score = tune_pipeline(
            pipeline, X, y, args.model, n_iter=args.n_iter, seed=args.seed
        )
        print(f"Param terbaik: {best_params} (CV Macro F1 {tune_score:.4f})")

    print(f"Cross-validation {args.cv}-fold:")
    cv = cross_validate_macro_f1(pipeline, X, y, args.cv, args.seed)
    print(f"Macro F1 CV: {cv['mean']:.4f} +/- {cv['std']:.4f}")

    weights, weighted_score = None, None
    if args.opt_weights:
        weights, weighted_score = optimize_class_weights(cv["oof_proba"], y, cv["classes"])
        print(f"Bobot kelas: {np.round(weights, 2).tolist()} -> Macro F1 OOF {weighted_score:.4f}")

    pred = cv["classes"][
        (cv["oof_proba"] * (weights if weights is not None else 1.0)).argmax(axis=1)
    ]
    print(classification_report(y, pred, digits=3, zero_division=0))

    print("Fit akhir pada seluruh data...")
    pipeline.fit(X, y)
    model_path = save_pipeline(pipeline, args.model, weights)
    print(f"Model disimpan: {model_path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "n_rows": int(len(X)),
        "cv_folds": args.cv,
        "fold_scores": cv["fold_scores"],
        "cv_macro_f1_mean": cv["mean"],
        "cv_macro_f1_std": cv["std"],
        "macro_f1_oof_with_class_weights": weighted_score,
        "class_weights": None if weights is None else [float(w) for w in weights],
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in best_params.items()},
    }
    (REPORT_DIR / f"cv_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print(f"Selesai dalam {time.time() - start:.0f} detik.")


if __name__ == "__main__":
    main()
