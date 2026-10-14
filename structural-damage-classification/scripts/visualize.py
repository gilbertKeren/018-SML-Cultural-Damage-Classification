"""Visualisasi EDA dan evaluasi model. Semua gambar disimpan ke ``reports/figures/``."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from scripts.feature_engineering import (
    BASE_NUMERIC_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    SUPERSTRUCTURE_COL,
    add_engineered_features,
    split_materials,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid")


def save_fig(name: str, fig_dir: Path | str | None = None, show: bool = True) -> Path:
    """Simpan figure yang sedang aktif lalu tampilkan.

    Parameters
    ----------
    name : str
        Nama file, mis. ``"target_distribution.png"``.
    fig_dir : str or pathlib.Path, optional
        Folder tujuan. Default ``reports/figures``.
    show : bool
        Jika ``False`` figure langsung ditutup (berguna untuk eksekusi di terminal).

    Returns
    -------
    pathlib.Path
        Lokasi file gambar.
    """
    fig_dir = Path(fig_dir) if fig_dir is not None else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    path = fig_dir / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close()
    return path


def plot_target_distribution(y: pd.Series, fig_dir=None) -> Path:
    """Bar chart distribusi ``damage_grade`` lengkap dengan persentase.

    Parameters
    ----------
    y : pandas.Series
        Target (1-5).
    fig_dir : optional
        Folder tujuan.
    """
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, color="#4C72B0", ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v / len(y):.1%}", ha="center", va="bottom")
    ax.set(title="Distribusi damage_grade", xlabel="damage_grade", ylabel="jumlah bangunan")
    return save_fig("target_distribution.png", fig_dir)


def plot_numeric_distributions(train: pd.DataFrame, fig_dir=None) -> Path:
    """Histogram empat fitur numerik (dipotong di persentil 99 agar outlier tidak mendominasi).

    Parameters
    ----------
    train : pandas.DataFrame
        Data latih.
    fig_dir : optional
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.ravel(), BASE_NUMERIC_COLS):
        s = train[col]
        sns.histplot(s[s <= s.quantile(0.99)], bins=40, color="#4C72B0", ax=ax)
        ax.set_title(f"{col} (<= persentil 99)")
    return save_fig("numeric_distributions.png", fig_dir)


def plot_age_vs_damage(train: pd.DataFrame, y: pd.Series, fig_dir=None) -> Path:
    """Boxplot ``log(1 + age_building)`` per ``damage_grade``.

    Parameters
    ----------
    train : pandas.DataFrame
    y : pandas.Series
    fig_dir : optional
    """
    age_df = pd.DataFrame(
        {"age_log": np.log1p(train["age_building"].clip(lower=0)), "damage_grade": y.values}
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=age_df, x="damage_grade", y="age_log", color="#8DA0CB", ax=ax)
    ax.set_title("log(1 + age_building) per damage_grade")
    return save_fig("age_vs_damage.png", fig_dir)


def plot_categorical_vs_damage(train: pd.DataFrame, y: pd.Series, fig_dir=None) -> Path:
    """Proporsi ``damage_grade`` pada setiap kategori fitur kategorikal.

    Parameters
    ----------
    train : pandas.DataFrame
    y : pandas.Series
    fig_dir : optional
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, col in zip(axes.ravel(), CATEGORICAL_COLS):
        ct = pd.crosstab(train[col].astype(str), y.values, normalize="index")
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.ravel()[len(CATEGORICAL_COLS):]:
        ax.axis("off")
    axes.ravel()[0].legend(title="grade", fontsize=8)
    return save_fig("categorical_vs_damage.png", fig_dir)


def plot_superstructure_vs_damage(train: pd.DataFrame, y: pd.Series, fig_dir=None, top_n: int = 10) -> Path:
    """Proporsi ``damage_grade`` untuk ``top_n`` material superstructure terbanyak.

    Parameters
    ----------
    train : pandas.DataFrame
    y : pandas.Series
    fig_dir : optional
    top_n : int
        Jumlah material yang ditampilkan.
    """
    tokens = train[SUPERSTRUCTURE_COL].fillna("unknown").astype(str).map(split_materials)
    sup = pd.DataFrame({"material": tokens, "grade": y.values}).explode("material")
    sup = sup.dropna().reset_index(drop=True)
    sup["material"] = sup["material"].str.replace("has_superstructure_", "", regex=False)
    top = sup["material"].value_counts().head(top_n).index
    sub = sup[sup["material"].isin(top)]
    ct = pd.crosstab(sub["material"], sub["grade"], normalize="index")
    order = ct.mul(range(1, ct.shape[1] + 1)).sum(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.loc[order].plot(kind="barh", stacked=True, colormap="viridis", ax=ax)
    ax.set(title=f"Proporsi damage_grade per material superstructure ({top_n} teratas)",
           xlabel="proporsi")
    ax.legend(title="grade", bbox_to_anchor=(1.01, 1), loc="upper left")
    return save_fig("superstructure_vs_damage.png", fig_dir)


def plot_damage_by_district(train: pd.DataFrame, y: pd.Series, fig_dir=None, top_n: int = 20) -> Path:
    """Heatmap proporsi ``damage_grade`` untuk ``top_n`` district dengan bangunan terbanyak.

    Parameters
    ----------
    train : pandas.DataFrame
    y : pandas.Series
    fig_dir : optional
    top_n : int
    """
    mask = train["district_id"].isin(train["district_id"].value_counts().head(top_n).index)
    ct = pd.crosstab(train.loc[mask, "district_id"], y.values[mask.values], normalize="index")
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "proporsi"}, ax=ax)
    ax.set_title("Proporsi damage_grade per district")
    return save_fig("damage_by_district.png", fig_dir)


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series, fig_dir=None) -> Path:
    """Heatmap korelasi Spearman fitur numerik (termasuk fitur turunan) dan target.

    Parameters
    ----------
    X : pandas.DataFrame
        Fitur mentah (tanpa target).
    y : pandas.Series
    fig_dir : optional
    """
    num_df = add_engineered_features(X)[NUMERIC_COLS].assign(damage_grade=y.values)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_df.corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title("Korelasi Spearman")
    return save_fig("correlation_heatmap.png", fig_dir)


def plot_model_comparison(comparison: pd.DataFrame, fig_dir=None) -> Path:
    """Bar horizontal perbandingan Macro F1 antar model.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Index = nama model; kolom ``macro_f1_mean`` dan ``macro_f1_std``.
    fig_dir : optional
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(comparison.index[::-1], comparison["macro_f1_mean"][::-1],
            xerr=comparison["macro_f1_std"][::-1], color="#4C72B0")
    ax.set(xlabel="Macro F1 (CV)", title="Perbandingan model")
    return save_fig("model_comparison.png", fig_dir)


def plot_confusion_matrix(y: pd.Series, pred: np.ndarray, fig_dir=None) -> Path:
    """Confusion matrix out-of-fold, dinormalisasi per baris (recall per kelas).

    Parameters
    ----------
    y : pandas.Series
        Label aktual.
    pred : numpy.ndarray
        Label prediksi.
    fig_dir : optional
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, normalize="true", cmap="Blues",
                                            values_format=".2f", ax=ax)
    ax.set_title("Confusion matrix OOF")
    return save_fig("confusion_matrix.png", fig_dir)


def plot_per_class_metrics(y: pd.Series, pred: np.ndarray, fig_dir=None) -> Path:
    """Bar chart precision, recall dan F1 untuk setiap kelas.

    Parameters
    ----------
    y : pandas.Series
    pred : numpy.ndarray
    fig_dir : optional
    """
    classes = [str(c) for c in np.sort(np.unique(y))]
    rep = pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T
    rep = rep.loc[classes, ["precision", "recall", "f1-score"]]
    ax = rep.plot(kind="bar", figsize=(8, 4), colormap="viridis")
    ax.set(title="Metrik per kelas (OOF)", xlabel="damage_grade", ylim=(0, 1))
    plt.xticks(rotation=0)
    return save_fig("per_class_metrics.png", fig_dir)


def plot_actual_vs_predicted(y: pd.Series, pred: np.ndarray, fig_dir=None) -> Path:
    """Bandingkan proporsi kelas aktual dengan proporsi kelas hasil prediksi.

    Parameters
    ----------
    y : pandas.Series
    pred : numpy.ndarray
    fig_dir : optional
    """
    dist = pd.DataFrame({
        "aktual": y.value_counts(normalize=True).sort_index(),
        "prediksi": pd.Series(pred).value_counts(normalize=True).sort_index(),
    })
    ax = dist.plot(kind="bar", figsize=(7, 4), color=["#4C72B0", "#DD8452"])
    ax.set(title="Distribusi kelas: aktual vs prediksi (OOF)", xlabel="damage_grade",
           ylabel="proporsi")
    plt.xticks(rotation=0)
    return save_fig("actual_vs_predicted_distribution.png", fig_dir)


def plot_feature_importance(
    pipeline, X: pd.DataFrame, y: pd.Series, seed: int = 42, n_rows: int = 15_000, fig_dir=None
) -> pd.Series:
    """Permutation importance (penurunan Macro F1) pada data hold-out.

    Pipeline di-clone dan dilatih pada 80% data, lalu setiap kolom mentah
    diacak pada hold-out untuk melihat seberapa besar skor turun.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline (dengan parameter terbaik); tidak perlu sudah di-fit.
    X, y : data latih (boleh subset).
    seed : int
    n_rows : int
        Jumlah baris hold-out yang dipakai untuk permutasi.
    fig_dir : optional

    Returns
    -------
    pandas.Series
        Importance per kolom, terurut naik.
    """
    X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    probe = clone(pipeline).fit(X_tr, y_tr)
    perm = permutation_importance(probe, X_ho.iloc[:n_rows], y_ho.iloc[:n_rows],
                                  scoring="f1_macro", n_repeats=3, random_state=seed, n_jobs=1)
    imp = pd.Series(perm.importances_mean, index=X_ho.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set(xlabel="Penurunan Macro F1 saat fitur diacak", title="Permutation importance")
    save_fig("feature_importance.png", fig_dir)
    return imp
