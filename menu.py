# -*- coding: utf-8 -*-
from dataclasses import dataclass

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
    out_tsv: str | None = None
    out_conllu: str | None = None
    out_xlsx: str | None = None
    out_json: str | None = None
    base_name: str | None = None
    # plots
    plots_from_tsv: str | None = None
    plots_out_dir: str | None = None
    wordcloud: bool | None = None
    topk_lemmas: int | None = None
    topk_deprel: int | None = None

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
    print("0) Salir")
    opt = input("\nElige una opción: ").strip()

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

    elif opt == "0":
        return MenuChoice(kind="exit")

    else:
        print("[!] Opción inválida.")
        return MenuChoice(kind="settings")
