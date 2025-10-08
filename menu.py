# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

@dataclass
class MenuConfig:
    input_path: str
    lang: str
    processors: str
    use_gpu: bool

@dataclass
class MenuChoice:
    kind: str
    # exportaciones
    out_tsv: Optional[str] = None
    out_conllu: Optional[str] = None
    out_xlsx: Optional[str] = None
    out_json: Optional[str] = None
    base_name: Optional[str] = None
    # plots
    plots_from_tsv: Optional[str] = None
    plots_out_dir: Optional[str] = None
    wordcloud: Optional[bool] = None
    topk_lemmas: Optional[int] = None
    topk_deprel: Optional[int] = None
    # stats
    stats_from_tsv: Optional[str] = None
    stats_top_lemmas: Optional[int] = None
    stats_window_cooc: Optional[int] = None
    stats_out_xlsx: Optional[str] = None

def print_header(cfg: MenuConfig):
    print("\n" + "=" * 60)
    print("            Análisis con Stanza – Menú principal")
    print("=" * 60)
    print(f"[Archivo ] {cfg.input_path}")
    print(f"[Idioma  ] {cfg.lang}")
    print(f"[Procesos] {cfg.processors}")
    print(f"[GPU     ] {'sí' if cfg.use_gpu else 'no'}")
    print("-" * 60)

def ask(prompt: str, default: str | None = None) -> str:
    s = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return s or (default or "")

def ask_int(prompt: str, default: int | None = None) -> int | None:
    raw = ask(prompt, str(default) if default is not None else None)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print("[!] Debe ser un número entero. Conservaré el valor por defecto.")
        return default

def ask_bool(prompt: str, default_yes: bool = False) -> bool:
    d = "s" if default_yes else "n"
    raw = ask(f"{prompt} (s/n)", d).lower()
    return raw.startswith("s")

def menu_loop(cfg: MenuConfig) -> MenuChoice:
    print_header(cfg)
    print("1) Ver análisis en consola (pretty print)")
    print("2) Exportar TSV (para Excel/CSV)")
    print("3) Exportar CoNLL-U")
    print("4) Exportar Excel (.xlsx) con resúmenes")
    print("5) Exportar TODO (pretty + .tsv + .conllu + .xlsx + .json)")
    print("6) Cambiar configuración (archivo, idioma, procesadores, GPU)")
    print("7) Generar gráficos (UPOS, lemas, NER, DEPREL, longitudes)")
    print("8) Calcular estadísticas y exportar a Excel (stats)")
    print("0) Salir  (también: q / quit / exit)")
    opt = input("\nElige una opción: ").strip().lower()

    if opt == "1":
        return MenuChoice(kind="pretty")

    elif opt == "2":
        name = ask("Nombre de salida TSV", "salida.tsv")
        return MenuChoice(kind="tsv", out_tsv=name)

    elif opt == "3":
        name = ask("Nombre de salida CoNLL-U", "salida.conllu")
        return MenuChoice(kind="conllu", out_conllu=name)

    elif opt == "4":
        name = ask("Nombre de salida Excel", "salida.xlsx")
        return MenuChoice(kind="xlsx", out_xlsx=name)

    elif opt == "5":
        base = ask("Base de nombre (sin extensión)", "salida")
        return MenuChoice(kind="all", base_name=base)

    elif opt == "6":
        # Cambiar configuración interactiva
        new_in = ask("Archivo de entrada", cfg.input_path) or cfg.input_path
        new_lang = ask("Idioma (es, en, fr...)", cfg.lang) or cfg.lang
        new_proc = ask("Procesadores Stanza (coma)", cfg.processors) or cfg.processors
        new_gpu = ask_bool("Usar GPU?", default_yes=cfg.use_gpu)

        cfg.input_path = new_in
        cfg.lang = new_lang
        cfg.processors = new_proc
        cfg.use_gpu = new_gpu
        return MenuChoice(kind="settings")

    elif opt == "7":
        # Plots desde un TSV ya generado
        tsv = ask("TSV de origen", "salida.tsv")
        out_dir = ask("Carpeta de salida de gráficos", "plots")
        wc = ask_bool("Incluir nube de palabras (wordcloud)?", default_yes=False)
        topk_lem = ask_int("Top-K lemas para gráfico", 20)
        topk_dep = ask_int("Top-K de relaciones DEPREL (vacío = todas)", None)

        return MenuChoice(
            kind="plots",
            plots_from_tsv=tsv,
            plots_out_dir=out_dir,
            wordcloud=wc,
            topk_lemmas=topk_lem,
            topk_deprel=topk_dep,
        )

    elif opt == "8":
        # Stats -> rama soportada en main.py
        tsv = ask("TSV de origen", "salida.tsv")
        top_lem = ask_int("Top lemas (entero)", 50)
        window = ask_int("Ventana de coocurrencia (entero)", 2)
        out_xlsx = ask("Archivo Excel de salida", "estadisticas.xlsx")
        return MenuChoice(
            kind="stats",
            stats_from_tsv=tsv,
            stats_top_lemmas=top_lem,
            stats_window_cooc=window,
            stats_out_xlsx=out_xlsx,
        )

    elif opt in {"0", "q", "quit", "exit"}:
        return MenuChoice(kind="exit")

    else:
        print("[!] Opción inválida. No se realizaron cambios.")
        return MenuChoice(kind="settings")

