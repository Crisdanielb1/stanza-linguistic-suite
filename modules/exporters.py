# -*- coding: utf-8 -*-
"""
exporters.py
-------------
Funciones de exportación para documentos de Stanza:
- TSV (token por fila)
- CoNLL-U (Universal Dependencies)
- Excel (.xlsx) con hojas de resumen
- JSON (tokens/entidades por oración)
"""

from __future__ import annotations
from pathlib import Path
from io import StringIO
from typing import Any, Dict, List

# Ajusta esta ruta a tu estructura real del proyecto
import Stanza.modules.stanza_demo as sd  # provee: doc_to_tsv, doc_to_conllu

# Dependencias opcionales (Excel)
try:
    import pandas as pd
except Exception as e:
    pd = None  # Solo necesario para Excel


# ===================== Helpers =====================

def ensure_suffix(p: Path, suffix: str) -> Path:
    """Añade la extensión si falta."""
    return p if p.suffix.lower() == suffix.lower() else p.with_suffix(suffix)


# ===================== Exportaciones base =====================

def export_tsv_from_doc(doc, out_path: Path) -> Path:
    """
    Exporta un Document de Stanza a TSV (token por fila).
    """
    out_path = ensure_suffix(Path(out_path), ".tsv")
    tsv_text = sd.doc_to_tsv(doc)
    out_path.write_text(tsv_text, encoding="utf-8")
    return out_path


def export_conllu_from_doc(doc, out_path: Path) -> Path:
    """
    Exporta un Document de Stanza a CoNLL-U.
    """
    out_path = ensure_suffix(Path(out_path), ".conllu")
    conllu_text = sd.doc_to_conllu(doc)
    out_path.write_text(conllu_text, encoding="utf-8")
    return out_path


def export_json_from_doc(doc, out_path: Path) -> Path:
    """
    Exporta un JSON simple con estructura {sentences: [...]} con tokens y NER.
    Útil para inspección o consumo por otras herramientas.
    """
    import json  # stdlib
    data: Dict[str, Any] = {"sentences": []}

    for s_ix, sent in enumerate(doc.sentences, start=1):
        sent_obj: Dict[str, Any] = {
            "index": s_ix,
            "text": sent.text,
            "tokens": [],
            "entities": [],
            "dependencies": [],
        }

        # tokens (usamos la primera word de cada token para lema/POS/DEP)
        for t_ix, tok in enumerate(sent.tokens, start=1):
            w = tok.words[0]
            sent_obj["tokens"].append({
                "tok_ix": t_ix,
                "form": tok.text,
                "lemma": w.lemma or "_",
                "upos": w.upos or "_",
                "xpos": w.xpos or "_",
                "head": w.head if w.head is not None else 0,
                "deprel": w.deprel or "_",
            })

        # dependencias (nivel word)
        id2form = {w.id: w.text for w in sent.words}
        for w in sent.words:
            head_form = "ROOT" if w.head in (0, None) else id2form.get(w.head, "ROOT")
            sent_obj["dependencies"].append({
                "head_form": head_form,
                "dep_form": w.text,
                "deprel": w.deprel,
                "head": w.head if w.head is not None else 0,
                "dep": w.id,
            })

        # entidades (si hay)
        if getattr(sent, "ents", None):
            for ent in sent.ents:
                sent_obj["entities"].append({
                    "type": ent.type,
                    "text": ent.text,
                    # Nota: los offsets exactos dependen del parser; aquí dejamos lo básico
                })

        data["sentences"].append(sent_obj)

    out_path = ensure_suffix(Path(out_path), ".json")
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ===================== Excel (con resúmenes) =====================

def _require_pandas():
    if pd is None:
        raise RuntimeError(
            "Para exportar a Excel necesitas instalar pandas y openpyxl:\n"
            "  pip install pandas openpyxl"
        )

def export_excel_from_tsv_text(tsv_text: str, out_path: Path) -> Path:
    """
    Convierte texto TSV (como el que genera doc_to_tsv) a un Excel con:
      - tokens (hoja principal)
      - pos_counts
      - lemma_freqs
      - ner_counts (si existe la columna NER)
    """
    _require_pandas()
    out_path = ensure_suffix(Path(out_path), ".xlsx")

    df = pd.read_csv(StringIO(tsv_text), sep="\t")
    pos_counts = df["UPOS"].value_counts().rename_axis("UPOS").reset_index(name="count")
    lemma_freqs = df["LEMMA"].value_counts().rename_axis("LEMMA").reset_index(name="count")
    ner_counts = (
        df["NER"].value_counts().rename_axis("NER_tag").reset_index(name="count")
        if "NER" in df.columns else
        pd.DataFrame(columns=["NER_tag", "count"])
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="tokens", index=False)
        pos_counts.to_excel(xw, sheet_name="pos_counts", index=False)
        lemma_freqs.to_excel(xw, sheet_name="lemma_freqs", index=False)
        ner_counts.to_excel(xw, sheet_name="ner_counts", index=False)

    return out_path


def export_excel_from_doc(doc, out_path: Path) -> Path:
    """
    Atajo: genera TSV desde el doc y luego exporta a Excel con resúmenes.
    """
    tsv_text = sd.doc_to_tsv(doc)
    return export_excel_from_tsv_text(tsv_text, out_path)


# ===================== Exportador “todo-en-uno” =====================

def export_all_from_doc(
    doc,
    base_path: Path,
    *,
    make_pretty_print: bool = False,
) -> List[Path]:
    """
    Exporta .tsv, .conllu, .xlsx y .json usando `base_path` como nombre base.
    Retorna la lista de rutas creadas.
    """
    created: List[Path] = []

    # TSV
    tsv_path = ensure_suffix(Path(base_path), ".tsv")
    export_tsv_from_doc(doc, tsv_path)
    created.append(tsv_path)

    # CONLLU
    conllu_path = ensure_suffix(Path(base_path), ".conllu")
    export_conllu_from_doc(doc, conllu_path)
    created.append(conllu_path)

    # XLSX
    xlsx_path = ensure_suffix(Path(base_path), ".xlsx")
    export_excel_from_doc(doc, xlsx_path)
    created.append(xlsx_path)

    # JSON
    json_path = ensure_suffix(Path(base_path), ".json")
    export_json_from_doc(doc, json_path)
    created.append(json_path)

    # Pretty print (solo muestra por consola; no crea archivo)
    if make_pretty_print:
        sd.print_pretty(doc)

    return created


__all__ = [
    "export_tsv_from_doc",
    "export_conllu_from_doc",
    "export_excel_from_doc",
    "export_excel_from_tsv_text",
    "export_json_from_doc",
    "export_all_from_doc",
    "ensure_suffix",
]
