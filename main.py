#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import argparse
import json

# ===================== Bootstrap para imports =====================
# Permite ejecutar el script desde cualquier ubicación:
#   - python -m Stanza.main
#   - python Stanza/main.py
PACKAGE_DIR = Path(__file__).resolve().parent           # .../Stanza
PROJECT_ROOT = PACKAGE_DIR.parent                       # .../Recursos_Intermedio
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ================================================================

# ===================== Imports consistentes =====================
import Stanza.modules.stanza_demo as sd
import Stanza.modules.exporters as ex
from Stanza.modules.utils import (
    load_json_config, save_json_config,
    read_text_smart, resolve_path, validate_processors,
)
# ================================================================
7
CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULT_LANG = "es"
DEFAULT_PROCS = "tokenize,mwt,pos,lemma,depparse,ner"
DEFAULT_INPUT = "mi_texto.txt"
DEFAULT_GPU = False

def load_config() -> dict:
    """Carga config.json con defaults; si no existe, lo crea."""
    defaults = {
        "lang": DEFAULT_LANG,
        "use_gpu": DEFAULT_GPU,
        "default_text": DEFAULT_INPUT,
        "processors": DEFAULT_PROCS,
    }
    return load_json_config(CONFIG_PATH, defaults)

def save_config(cfg: dict) -> None:
    """Guarda configuración filtrando claves permitidas."""
    allowed = {"lang", "use_gpu", "default_text", "processors"}
    save_json_config(CONFIG_PATH, cfg, allowed_keys=allowed)

# ===================== Utilidades =====================

def read_input_text(path_str: str) -> str:
    """
    Lee el archivo de entrada intentando:
      1) ruta tal cual / CWD
      2) relativo al directorio de este script
    """
    p = Path(path_str)
    if not p.exists():
        p = resolve_path(path_str, base=Path(__file__).parent)
    if not p.exists():
        sys.exit(f"[x] No se encontró el archivo de entrada: {p.resolve()}")
    return read_text_smart(str(p))

# ===================== Núcleo =====================

def run_analysis(
    input_path: str,
    lang: str = DEFAULT_LANG,
    processors: str = DEFAULT_PROCS,
    use_gpu: bool = False,
    do_pretty: bool = False,
    out_tsv: Path | None = None,
    out_conllu: Path | None = None,
    out_xlsx: Path | None = None,
    out_json: Path | None = None,   # opcional extra
):
    """Ejecuta el pipeline de Stanza y produce las salidas solicitadas."""
    # 1) Texto
    text = read_input_text(input_path)

    # 2) Normalizar/validar cadena de procesadores
    proc_norm, warns = validate_processors(processors)
    for w in warns:
        print(f"[!] {w}")
    if proc_norm != processors:
        print(f"[i] Procesadores normalizados: {processors} -> {proc_norm}")
    processors = proc_norm

    # 3) Pipeline (descarga modelo si falta)
    nlp = sd.build_pipeline(lang=lang, processors=processors, use_gpu=use_gpu)
    doc = nlp(text)

    # 4) Consola bonita (opcional)
    if do_pretty:
        sd.print_pretty(doc)

    # 5) Exportaciones (centralizadas)
    if out_tsv:
        ex.export_tsv_from_doc(doc, Path(out_tsv))
        print(f"[✓] TSV guardado en: {ex.ensure_suffix(Path(out_tsv), '.tsv')}")

    if out_conllu:
        ex.export_conllu_from_doc(doc, Path(out_conllu))
        print(f"[✓] CoNLL-U guardado en: {ex.ensure_suffix(Path(out_conllu), '.conllu')}")

    if out_xlsx:
        ex.export_excel_from_doc(doc, Path(out_xlsx))
        print(f"[✓] Excel guardado en: {ex.ensure_suffix(Path(out_xlsx), '.xlsx')}")

    if out_json:
        ex.export_json_from_doc(doc, Path(out_json))
        print(f"[✓] JSON guardado en: {ex.ensure_suffix(Path(out_json), '.json')}")

def main():
    # Import tardío del menú (con fallback)
    try:
        from .menu import menu_loop, MenuConfig, MenuChoice
    except Exception:
        from menu import menu_loop, MenuConfig, MenuChoice

    # 1) Cargar config persistente
    cfg_json = load_config()

    # 2) CLI con defaults desde config.json
    parser = argparse.ArgumentParser(
        description="Frontend del proyecto Stanza: menú interactivo o modo rápido."
    )
    parser.add_argument("-i", "--input", default=cfg_json.get("default_text", DEFAULT_INPUT),
                        help="Archivo de entrada (texto plano).")
    parser.add_argument("-l", "--lang", default=cfg_json.get("lang", DEFAULT_LANG),
                        help="Idioma Stanza (ej. es, en...).")
    parser.add_argument("-p", "--processors", default=cfg_json.get("processors", DEFAULT_PROCS),
                        help="Procesadores de Stanza (coma).")
    parser.add_argument("--gpu", action="store_true", default=cfg_json.get("use_gpu", DEFAULT_GPU),
                        help="Usar GPU si está disponible.")
    parser.add_argument("--quick-tsv", action="store_true",
                        help="Modo rápido: genera salida.tsv (sin menú).")
    args = parser.parse_args()

    # 3) Actualizar y guardar config con lo que vino por CLI (último estado)
    cfg_json.update({
        "default_text": args.input,
        "lang": args.lang,
        "processors": args.processors,
        "use_gpu": bool(args.gpu),
    })
    save_config(cfg_json)

    # 4) Modo rápido: solo TSV
    if args.quick_tsv:
        run_analysis(
            input_path=args.input,
            lang=args.lang,
            processors=args.processors,
            use_gpu=args.gpu,
            do_pretty=False,
            out_tsv=Path("salida.tsv"),
        )
        return

    # 5) Modo interactivo (menú)
    cfg = MenuConfig(
        input_path=args.input,
        lang=args.lang,
        processors=args.processors,
        use_gpu=args.gpu,
    )

    while True:
        choice = menu_loop(cfg)

        if choice.kind == "exit":
            print("¡Listo! Hasta luego 👋")
            break

        elif choice.kind == "pretty":
            run_analysis(
                input_path=cfg.input_path,
                lang=cfg.lang,
                processors=cfg.processors,
                use_gpu=cfg.use_gpu,
                do_pretty=True,
            )

        elif choice.kind == "tsv":
            out = Path(choice.out_tsv or "salida.tsv")
            run_analysis(
                input_path=cfg.input_path,
                lang=cfg.lang,
                processors=cfg.processors,
                use_gpu=cfg.use_gpu,
                out_tsv=out,
            )

        elif choice.kind == "conllu":
            out = Path(choice.out_conllu or "salida.conllu")
            run_analysis(
                input_path=cfg.input_path,
                lang=cfg.lang,
                processors=cfg.processors,
                use_gpu=cfg.use_gpu,
                out_conllu=out,
            )

        elif choice.kind == "xlsx":
            out = Path(choice.out_xlsx or "salida.xlsx")
            run_analysis(
                input_path=cfg.input_path,
                lang=cfg.lang,
                processors=cfg.processors,
                use_gpu=cfg.use_gpu,
                out_xlsx=out,
            )

        elif choice.kind == "all":
            base = Path(choice.base_name or "salida")
            run_analysis(
                input_path=cfg.input_path,
                lang=cfg.lang,
                processors=cfg.processors,
                use_gpu=cfg.use_gpu,
                do_pretty=True,
                out_tsv=base.with_suffix(".tsv"),
                out_conllu=base.with_suffix(".conllu"),
                out_xlsx=base.with_suffix(".xlsx"),
                out_json=base.with_suffix(".json"),
            )

        elif choice.kind == "plots":
            # Generación de gráficos desde un TSV existente
            try:
                from .modules import plots as pl
            except Exception:
                import Stanza.modules.plots as pl
            outs = pl.generate_all_plots(
                tsv_path=choice.plots_from_tsv or "salida.tsv",
                out_dir=choice.plots_out_dir or "plots",
                topk_lemmas=choice.topk_lemmas or 20,
                topk_deprel=choice.topk_deprel,           # puede ser None
                make_wordcloud=bool(choice.wordcloud),    # True/False
                stopwords=None,                           # opcional: set({...})
            )
            print("[✓] Gráficos guardados:")
            for k, v in outs.items():
                print(f"  - {k}: {v}")

        elif choice.kind == "settings":
            # Guardar configuración persistente cuando cambie desde el menú
            save_config({
                "default_text": cfg.input_path,
                "lang": cfg.lang,
                "processors": cfg.processors,
                "use_gpu": cfg.use_gpu,
            })
            print("[✓] Configuración actualizada y guardada en config.json.")

        elif choice.kind == "stats":
            # Exporta estadísticas a Excel desde un TSV
            try:
                from .modules import stats as st
            except Exception:
                from Stanza.modules import stats as st
            pack = st.build_all_stats(
                tsv_path=choice.stats_from_tsv or "salida.tsv",
                top_lemmas=choice.stats_top_lemmas or 50,
                window_cooc=choice.stats_window_cooc or 2,
            )
            out_xlsx = choice.stats_out_xlsx or "estadisticas.xlsx"
            path = st.export_stats_to_excel(pack, out_xlsx)
            print(f"[✓] Estadísticas exportadas a: {path}")

        else:
            print("[!] Opción no reconocida.")

if __name__ == "__main__":
    main()
