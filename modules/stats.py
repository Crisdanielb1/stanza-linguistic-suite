# -*- coding: utf-8 -*-
"""
stats.py
--------
Cálculo de estadísticas a partir del TSV (token por fila) generado por Stanza.

Columnas esperadas en el TSV:
- sent_ix, tok_ix, FORM, LEMMA, UPOS, XPOS, HEAD, DEPREL, NER

Funciones principales:
- load_tsv
- pos_counts, lemma_freqs, ner_counts, deprel_counts
- top_lemmas_by_upos
- sentence_lengths
- cooccurrences_within_window
- dependency_role_matrix
- build_all_stats
- export_stats_to_excel
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, Dict

import pandas as pd


# ===================== Lectura / normalización =====================

def load_tsv(tsv_path: Path | str) -> pd.DataFrame:
    """Carga TSV y normaliza algunos tipos / valores nulos."""
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"No se encontró el TSV: {tsv_path.resolve()}")

    df = pd.read_csv(tsv_path, sep="\t")
    # Normalizaciones
    for col, default in [
        ("UPOS", "_"),
        ("LEMMA", "_"),
        ("NER", "O"),
        ("DEPREL", "_"),
        ("FORM", "_"),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(str)

    # Tipos numéricos
    if "sent_ix" in df.columns:
        df["sent_ix"] = pd.to_numeric(df["sent_ix"], errors="coerce").astype("Int64")
    if "tok_ix" in df.columns:
        df["tok_ix"] = pd.to_numeric(df["tok_ix"], errors="coerce").astype("Int64")
    if "HEAD" in df.columns:
        df["HEAD"] = pd.to_numeric(df["HEAD"], errors="coerce").fillna(0).astype(int)

    return df


# ===================== Estadísticas base =====================

def pos_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "UPOS" not in df.columns:
        raise ValueError("Falta columna 'UPOS'.")
    s = df["UPOS"].value_counts()
    out = s.rename_axis("UPOS").reset_index(name="count")
    out["percent"] = (out["count"] / out["count"].sum() * 100).round(2)
    return out

def lemma_freqs(df: pd.DataFrame, top: int = 100) -> pd.DataFrame:
    if "LEMMA" not in df.columns:
        raise ValueError("Falta columna 'LEMMA'.")
    s = df["LEMMA"].value_counts().head(top)
    return s.rename_axis("LEMMA").reset_index(name="count")

def ner_counts(df: pd.DataFrame, drop_o: bool = True) -> pd.DataFrame:
    if "NER" not in df.columns:
        raise ValueError("Falta columna 'NER'.")
    s = df["NER"]
    if drop_o:
        s = s[s != "O"]
    s = s.value_counts()
    return s.rename_axis("NER_tag").reset_index(name="count")

def deprel_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "DEPREL" not in df.columns:
        raise ValueError("Falta columna 'DEPREL'.")
    s = df["DEPREL"].value_counts()
    return s.rename_axis("DEPREL").reset_index(name="count")


# ===================== Estadísticas derivadas =====================

def top_lemmas_by_upos(df: pd.DataFrame, upos: str, top: int = 20) -> pd.DataFrame:
    """Top lemas para una categoría UPOS específica."""
    for col in ("UPOS", "LEMMA"):
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}'.")
    sub = df[df["UPOS"] == upos]
    s = sub["LEMMA"].value_counts().head(top)
    return s.rename_axis("LEMMA").reset_index(name="count")

def sentence_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Tokens por oración usando sent_ix."""
    if "sent_ix" not in df.columns:
        raise ValueError("Falta columna 'sent_ix'.")
    lengths = df.groupby("sent_ix")["FORM"].count().rename("length")
    return lengths.reset_index()

def cooccurrences_within_window(
    df: pd.DataFrame,
    window: int = 2,
    *,
    by_sent: bool = True,
    lowercase: bool = True,
    exclude: Optional[Iterable[str]] = None,
    use_lemma: bool = True,
    top: Optional[int] = 200,
) -> pd.DataFrame:
    """
    Coocurrencias simétricas dentro de una ventana +-k por sentencia.
    Retorna columnas: left, right, count (left<=right para evitar duplicados).
    """
    if "FORM" not in df.columns or "LEMMA" not in df.columns:
        raise ValueError("Faltan columnas 'FORM' / 'LEMMA'.")

    token_col = "LEMMA" if use_lemma else "FORM"
    x = df[[ "sent_ix", "tok_ix", token_col ]].copy()
    x[token_col] = x[token_col].astype(str)
    if lowercase:
        x[token_col] = x[token_col].str.lower()
    if exclude:
        excl = {e.lower() if lowercase else e for e in exclude}
        x = x[~x[token_col].isin(excl)]

    # Recorremos por oración
    pairs: Dict[tuple, int] = {}
    if by_sent:
        groups = x.groupby("sent_ix", dropna=True)
    else:
        # todo el doc como una sola secuencia
        groups = [(0, x.sort_values(["sent_ix", "tok_ix"]))]

    for _, g in groups:
        g = g.sort_values("tok_ix")
        tokens = g[token_col].tolist()
        n = len(tokens)
        for i in range(n):
            a = tokens[i]
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j == i:
                    continue
                b = tokens[j]
                left, right = sorted((a, b))
                pairs[(left, right)] = pairs.get((left, right), 0) + 1

    rows = [{"left": k[0], "right": k[1], "count": v} for k, v in pairs.items()]
    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    if top:
        out = out.head(top)
    return out.reset_index(drop=True)

def dependency_role_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matriz DEPREL x UPOS que indica cuántas veces un token de cierta UPOS
    aparece con una relación de dependencia determinada.
    """
    for col in ("DEPREL", "UPOS"):
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}'.")
    ctab = pd.crosstab(df["DEPREL"], df["UPOS"])
    ctab = ctab.sort_index()
    # Añade porcentajes por fila
    ctab_pct = (ctab.div(ctab.sum(axis=1), axis=0) * 100).round(2)
    # Empaquetar en un MultiIndex amigable (opcional)
    ctab_pct.index.name = "DEPREL"
    ctab_pct.columns.name = "UPOS"
    return ctab_pct


# ===================== Paquetes de resultados =====================

def build_all_stats(
    tsv_path: Path | str,
    *,
    top_lemmas: int = 50,
    window_cooc: int = 2,
    exclude_tokens: Optional[Iterable[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Calcula un paquete de estadísticas a partir del TSV.
    Retorna un dict con varios DataFrames: pos, lemmas, ner, deprel, lengths, cooc, dep_matrix.
    """
    df = load_tsv(tsv_path)

    stats: dict[str, pd.DataFrame] = {}
    stats["pos"] = pos_counts(df)
    stats["lemmas"] = lemma_freqs(df, top=top_lemmas)
    try:
        stats["ner"] = ner_counts(df, drop_o=True)
    except ValueError:
        stats["ner"] = pd.DataFrame(columns=["NER_tag", "count"])
    stats["deprel"] = deprel_counts(df)
    stats["lengths"] = sentence_lengths(df)
    stats["cooc"] = cooccurrences_within_window(
        df,
        window=window_cooc,
        exclude=exclude_tokens,
        use_lemma=True,
    )
    stats["dep_matrix_pct"] = dependency_role_matrix(df)
    return stats


def export_stats_to_excel(
    stats: dict[str, pd.DataFrame],
    out_xlsx: Path | str = "stats.xlsx",
) -> Path:
    """
    Exporta el paquete de estadísticas a un Excel (cada clave del dict = una hoja).
    """
    out_xlsx = Path(out_xlsx)
    if out_xlsx.suffix.lower() != ".xlsx":
        out_xlsx = out_xlsx.with_suffix(".xlsx")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        for sheet, data in stats.items():
            # Asegurar DataFrame
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            # Limitar nombres de hoja a 31 chars (Excel)
            sheet_name = sheet[:31] if sheet else "sheet"
            df.to_excel(xw, sheet_name=sheet_name, index=True if df.index.name else False)

    return out_xlsx


__all__ = [
    "load_tsv",
    "pos_counts",
    "lemma_freqs",
    "ner_counts",
    "deprel_counts",
    "top_lemmas_by_upos",
    "sentence_lengths",
    "cooccurrences_within_window",
    "dependency_role_matrix",
    "build_all_stats",
    "export_stats_to_excel",
]
