#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path
import stanza

def build_pipeline(lang: str, processors: str, use_gpu: bool):
    # Descarga modelos si no están
    stanza.download(lang, processors=processors, verbose=False)
    return stanza.Pipeline(
        lang=lang,
        processors=processors,
        use_gpu=use_gpu,
        tokenize_pretokenized=False,
        verbose=False,
    )

def doc_to_conllu(doc: stanza.Document) -> str:
    lines = []
    sent_id = 0
    for sent in doc.sentences:
        sent_id += 1
        lines.append(f"# sent_id = {sent_id}")
        lines.append(f"# text = {sent.text}")
        for w in sent.words:
            # Campos CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            feats = w.feats if w.feats else "_"
            xpos = w.xpos if w.xpos else "_"
            misc = "_"
            deps = "_"
            lines.append("\t".join(map(str, [
                w.id, w.text, w.lemma, w.upos, xpos, feats,
                w.head if w.head is not None else 0, w.deprel, deps, misc
            ])))
        lines.append("")  # línea en blanco separadora
    return "\n".join(lines).rstrip() + "\n"

def doc_to_tsv(doc: stanza.Document) -> str:
    rows = ["sent_ix\ttok_ix\tFORM\tLEMMA\tUPOS\tXPOS\tHEAD\tDEPREL\tNER"]
    for s_ix, sent in enumerate(doc.sentences, start=1):
        # BIO por token, por defecto "O"
        bio_tags = ["O"] * len(sent.tokens)

        # Mapa: token (objeto) -> índice de token en la oración (0..n-1)
        token_positions = {tok: idx for idx, tok in enumerate(sent.tokens)}

        # Marcar entidades en BIO usando posiciones de tokens
        if getattr(sent, "ents", None):
            for ent in sent.ents:
                toks = getattr(ent, "tokens", None)
                if toks:
                    # obtener índices dentro de la oración
                    idxs = sorted(
                        [token_positions[t] for t in toks if t in token_positions]
                    )
                    if idxs:
                        bio_tags[idxs[0]] = f"B-{ent.type}"
                        for j in idxs[1:]:
                            bio_tags[j] = f"I-{ent.type}"

        # Filas token por token (usamos la primera word del token para lema/POS/DEP)
        for t_ix, (tok, bio) in enumerate(zip(sent.tokens, bio_tags), start=1):
            w = tok.words[0]
            rows.append("\t".join(map(str, [
                s_ix,
                t_ix,
                tok.text,
                w.lemma or "_",
                w.upos or "_",
                w.xpos or "_",
                w.head if w.head is not None else 0,
                w.deprel or "_",
                bio
            ])))
    return "\n".join(rows) + "\n"


def print_pretty(doc: stanza.Document):
    for i, sent in enumerate(doc.sentences, start=1):
        print(f"\n=== Oración {i} ===")
        print(sent.text)
        print("\nTokens / Lema / UPOS")
        print("-" * 28)
        for w in sent.words:
            print(f"{w.text:>15}  {w.lemma or '_':>15}  {w.upos or '_':>6}")

        # Árbol de dependencias (cabeza -> dependiente)
        print("\nDependencias (HEAD ─deprel→ DEP)")
        print("-" * 28)
        # mapear ids a formas para mostrar cabezas
        id2form = {w.id: w.text for w in sent.words}
        for w in sent.words:
            head_form = "ROOT" if w.head in (0, None) else id2form.get(w.head, "ROOT")
            print(f"{head_form} ─{w.deprel}→ {w.text}")

        # Entidades
        if getattr(sent, "ents", []):
            print("\nEntidades NER:")
            for ent in sent.ents:
                print(f"[{ent.type}] «{ent.text}»")
        else:
            print("\nEntidades NER: (ninguna)")

def main():
    parser = argparse.ArgumentParser(
        description="Demo mínima de Stanza: tokenización, POS, lema, dependencias y NER."
    )
    parser.add_argument("-l", "--lang", default="es",
                        help="Idioma (por ej. es, en, fr). Default: es")
    parser.add_argument("-p", "--processors",
                        default="tokenize,mwt,pos,lemma,depparse,ner",
                        help="Lista de procesadores de Stanza (coma).")
    parser.add_argument("--gpu", action="store_true",
                        help="Usar GPU si está disponible.")
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Ruta a archivo de texto. Si se omite, lee de stdin o usa ejemplo.")
    parser.add_argument("--save-conllu", type=str, default=None,
                        help="Ruta para guardar salida en CoNLL-U.")
    parser.add_argument("--save-tsv", type=str, default=None,
                        help="Ruta para guardar tabla TSV (token por fila).")
    args = parser.parse_args()

    # Texto de entrada
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            # Ejemplo corto (ES) / (EN) según idioma
            text = ("Ayer hablé con María en San José sobre el proyecto LESCO. "
                    "Quedamos en enviar el informe el viernes.") if args.lang.startswith("es") \
                   else ("Yesterday I spoke with Maria in San Jose about the LESCO project. "
                         "We agreed to send the report on Friday.")

    nlp = build_pipeline(args.lang, args.processors, args.gpu)
    doc = nlp(text)

    # Consola bonita
    print_pretty(doc)

    # Guardados opcionales
    if args.save_conllu:
        Path(args.save_conllu).write_text(doc_to_conllu(doc), encoding="utf-8")
        print(f"\n[✓] CoNLL-U guardado en: {args.save_conllu}")
    if args.save_tsv:
        Path(args.save_tsv).write_text(doc_to_tsv(doc), encoding="utf-8")
        print(f"[✓] TSV guardado en: {args.save_tsv}")

if __name__ == "__main__":
    main()
