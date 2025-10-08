# -*- coding: utf-8 -*-
"""
plots.py
--------
Gráficos básicos para explorar la salida TSV del pipeline (token por fila):
- Frecuencia por UPOS
- Top-K lemas
- Frecuencia por etiqueta NER (BIO)
- Frecuencia por relación de dependencia (DEPREL)
- Longitud de oraciones (tokens por oración)
- (Opcional) Nube de palabras por lemmas

NOTAS:
- Usa matplotlib (sin seaborn), una figura por gráfico.
- No define colores explícitos (usa los que vengan por defecto).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


# ===================== Helpers =====================

def _ensure_output_dir(out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path

def _ensure_suffix(p: Path, suffix: str) -> Path:
    return p if p.suffix.lower() == suffix.lower() else p.with_suffix(suffix)

def _load_tsv(tsv_path: Path | str) -> pd.DataFrame:
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"No se encontró el TSV: {tsv_path.resolve()}")
    df = pd.read_csv(tsv_path, sep="\t")
    # Normalizaciones suaves
    if "UPOS" in df.columns:
        df["UPOS"] = df["UPOS"].fillna("_").astype(str)
    if "LEMMA" in df.columns:
        df["LEMMA"] = df["LEMMA"].fillna("_").astype(str)
    if "NER" in df.columns:
        df["NER"] = df["NER"].fillna("O").astype(str)
    if "DEPREL" in df.columns:
        df["DEPREL"] = df["DEPREL"].fillna("_").astype(str)
    return df


# ===================== Plots =====================

def plot_pos_counts(
    tsv_path: Path | str,
    out_path: Path | str,
    *,
    sort_desc: bool = True,
    rotate_xticks: int = 45,
) -> Path:
    """
    Barras de frecuencia por UPOS.
    """
    df = _load_tsv(tsv_path)
    if "UPOS" not in df.columns:
        raise ValueError("El TSV no contiene la columna 'UPOS'.")
    counts = df["UPOS"].value_counts(ascending=not sort_desc)

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Frecuencia por categoría gramatical (UPOS)")
    plt.xlabel("UPOS")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=rotate_xticks)
    plt.tight_layout()

    out_path = _ensure_suffix(Path(out_path), ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_lemma_topk(
    tsv_path: Path | str,
    out_path: Path | str,
    *,
    topk: int = 20,
    rotate_xticks: int = 45,
) -> Path:
    """
    Barras con los Top-K lemas más frecuentes.
    """
    df = _load_tsv(tsv_path)
    if "LEMMA" not in df.columns:
        raise ValueError("El TSV no contiene la columna 'LEMMA'.")
    counts = df["LEMMA"].value_counts().head(topk)

    plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Lemas más frecuentes (Top-{topk})")
    plt.xlabel("LEMMA")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    out_path = _ensure_suffix(Path(out_path), ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_ner_counts(
    tsv_path: Path | str,
    out_path: Path | str,
    *,
    drop_o: bool = True,
    rotate_xticks: int = 45,
) -> Path:
    """
    Barras de frecuencia por etiqueta NER (BIO).
    Por defecto elimina la clase 'O' (no-entidad).
    """
    df = _load_tsv(tsv_path)
    if "NER" not in df.columns:
        raise ValueError("El TSV no contiene la columna 'NER'.")

    ner_series = df["NER"]
    if drop_o:
        ner_series = ner_series[ner_series != "O"]

    counts = ner_series.value_counts()
    if counts.empty:
        # Graficar placeholder vacío
        plt.figure()
        plt.title("Frecuencia NER (sin entidades)")
        plt.text(0.5, 0.5, "No hay etiquetas NER distintas de 'O'", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
    else:
        plt.figure()
        counts.plot(kind="bar")
        plt.title("Frecuencia por etiqueta NER (BIO)")
        plt.xlabel("NER tag")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=rotate_xticks)
        plt.tight_layout()

    out_path = _ensure_suffix(Path(out_path), ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_deprel_counts(
    tsv_path: Path | str,
    out_path: Path | str,
    *,
    topk: Optional[int] = None,
    rotate_xticks: int = 60,
) -> Path:
    """
    Barras de frecuencia por relación de dependencia (DEPREL).
    Si topk se especifica, muestra solo las top-k relaciones.
    """
    df = _load_tsv(tsv_path)
    if "DEPREL" not in df.columns:
        raise ValueError("El TSV no contiene la columna 'DEPREL'.")
    counts = df["DEPREL"].value_counts()
    if topk:
        counts = counts.head(topk)

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Frecuencia por relación de dependencia (DEPREL)")
    plt.xlabel("DEPREL")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    out_path = _ensure_suffix(Path(out_path), ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_sentence_lengths(
    tsv_path: Path | str,
    out_path: Path | str,
    *,
    bins: int = 10,
) -> Path:
    """
    Histograma de longitudes de oración (tokens por oración) usando 'sent_ix'.
    """
    df = _load_tsv(tsv_path)
    if "sent_ix" not in df.columns:
        raise ValueError("El TSV no contiene la columna 'sent_ix'.")
    lengths = df.groupby("sent_ix")["FORM"].count()

    plt.figure()
    plt.hist(lengths, bins=bins)
    plt.title("Distribución de longitudes de oración (tokens por oración)")
    plt.xlabel("Tokens por oración")
    plt.ylabel("Frecuencia")
    plt.tight_layout()

    out_path = _ensure_suffix(Path(out_path), ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ===================== Nube de palabras (opcional) =====================

def plot_wordcloud_lemmas(
    tsv_path: Path | str,
    out_path: Path | str,
    *,
    min_length: int = 2,
    stopwords: Optional[set[str]] = None,
    width: int = 1200,
    height: int = 600,
) -> Path:
    """
    Nube de palabras por LEMMA (requiere `wordcloud`).
    Filtra lemas muy cortos y stopwords si se proveen.
    """
    try:
        from wordcloud import WordCloud
    except Exception as e:
        raise RuntimeError("Para la nube de palabras instala 'wordcloud': pip install wordcloud") from e

    df = _load_tsv(tsv_path)
    if "LEMMA" not in df.columns:
        raise ValueError("El TSV no contiene la columna 'LEMMA'.")

    lemmas = df["LEMMA"].astype(str)
    if stopwords:
        lemmas = lemmas[~lemmas.str.lower().isin({w.lower() for w in stopwords})]
    if min_length > 1:
        lemmas = lemmas[lemmas.str.len() >= min_length]

    text = " ".join(lemmas.tolist())
    wc = WordCloud(width=width, height=height, background_color="white").generate(text)

    out_path = _ensure_suffix(Path(out_path), ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(out_path))
    return out_path


# ===================== Pipeline de conveniencia =====================

def generate_all_plots(
    tsv_path: Path | str,
    out_dir: Path | str = "plots",
    *,
    topk_lemmas: int = 20,
    topk_deprel: Optional[int] = None,
    make_wordcloud: bool = False,
    stopwords: Optional[set[str]] = None,
) -> dict[str, Path]:
    """
    Genera todos los gráficos estándar y retorna un dict con las rutas creadas.
    """
    out_dir = _ensure_output_dir(out_dir)
    tsv_path = Path(tsv_path)

    outputs: dict[str, Path] = {}

    outputs["upos"] = plot_pos_counts(tsv_path, out_dir / "upos_counts.png")
    outputs["lemmas"] = plot_lemma_topk(tsv_path, out_dir / f"lemma_top{topk_lemmas}.png", topk=topk_lemmas)
    outputs["ner"] = plot_ner_counts(tsv_path, out_dir / "ner_counts.png")
    outputs["deprel"] = plot_deprel_counts(tsv_path, out_dir / "deprel_counts.png", topk=topk_deprel)
    outputs["sentlen"] = plot_sentence_lengths(tsv_path, out_dir / "sentence_lengths.png")

    if make_wordcloud:
        outputs["wordcloud"] = plot_wordcloud_lemmas(tsv_path, out_dir / "lemma_wordcloud.png", stopwords=stopwords)

    return outputs


__all__ = [
    "plot_pos_counts",
    "plot_lemma_topk",
    "plot_ner_counts",
    "plot_deprel_counts",
    "plot_sentence_lengths",
    "plot_wordcloud_lemmas",
    "generate_all_plots",
]
