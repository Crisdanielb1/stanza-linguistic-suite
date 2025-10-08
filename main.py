#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import threading
import queue

# ===================== Bootstrap para imports =====================
from pathlib import Path as _Path
PACKAGE_DIR = _Path(__file__).resolve().parent           # .../Stanza
PROJECT_ROOT = PACKAGE_DIR.parent                        # .../Recursos_Intermedio
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

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULT_LANG = "es"
DEFAULT_PROCS = "tokenize,mwt,pos,lemma,depparse,ner"
DEFAULT_INPUT = "mi_texto.txt"
DEFAULT_GPU = False


# ===================== Configuración =====================

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
    log_fn = print,                 # para GUI: inyectar logger
):
    """Ejecuta el pipeline de Stanza y produce las salidas solicitadas."""
    # 1) Texto
    text = read_input_text(input_path)

    # 2) Normalizar/validar cadena de procesadores
    proc_norm, warns = validate_processors(processors)
    for w in warns:
        log_fn(f"[!] {w}")
    if proc_norm != processors:
        log_fn(f"[i] Procesadores normalizados: {processors} -> {proc_norm}")
    processors = proc_norm

    # 3) Pipeline (descarga modelo si falta)
    log_fn("[i] Construyendo pipeline…")
    nlp = sd.build_pipeline(lang=lang, processors=processors, use_gpu=use_gpu)
    doc = nlp(text)

    # 4) Consola bonita (opcional)
    if do_pretty:
        log_fn("[i] Impresión bonita del documento:")
        sd.print_pretty(doc)

    # 5) Exportaciones
    if out_tsv:
        ex.export_tsv_from_doc(doc, Path(out_tsv))
        log_fn(f"[✓] TSV guardado en: {ex.ensure_suffix(Path(out_tsv), '.tsv')}")

    if out_conllu:
        ex.export_conllu_from_doc(doc, Path(out_conllu))
        log_fn(f"[✓] CoNLL-U guardado en: {ex.ensure_suffix(Path(out_conllu), '.conllu')}")

    if out_xlsx:
        ex.export_excel_from_doc(doc, Path(out_xlsx))
        log_fn(f"[✓] Excel guardado en: {ex.ensure_suffix(Path(out_xlsx), '.xlsx')}")

    if out_json:
        ex.export_json_from_doc(doc, Path(out_json))
        log_fn(f"[✓] JSON guardado en: {ex.ensure_suffix(Path(out_json), '.json')}")


# ===================== GUI (Tkinter) =====================

def launch_gui(cfg_init: dict):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    log_queue: "queue.Queue[str]" = queue.Queue()

    def gui_log(msg: str):
        log_queue.put(msg)

    def drain_log():
        try:
            while True:
                msg = log_queue.get_nowait()
                txt_log.configure(state="normal")
                txt_log.insert("end", msg + "\n")
                txt_log.see("end")
                txt_log.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(120, drain_log)

    def browse_input():
        path = filedialog.askopenfilename(
            title="Seleccionar archivo de texto",
            filetypes=[("Texto", "*.txt *.md *.csv *.tsv *.conllu *.json"), ("Todos", "*.*")]
        )
        if path:
            var_input.set(path)

    # --- NUEVO: exploradores para TSV de Plots/Stats ---
    def browse_plots_tsv():
        p = filedialog.askopenfilename(
            title="Seleccionar TSV para gráficos",
            filetypes=[("TSV", "*.tsv"), ("Todos", "*.*")]
        )
        if p:
            var_plots_tsv.set(p)

    def browse_stats_tsv():
        p = filedialog.askopenfilename(
            title="Seleccionar TSV para estadísticas",
            filetypes=[("TSV", "*.tsv"), ("Todos", "*.*")]
        )
        if p:
            var_stats_tsv.set(p)

    # Validación: si escriben un TSV que no existe, pedir localizar
    def ensure_tsv_exists(var: tk.StringVar, browse_fn):
        path = Path(var.get().strip())
        if not path.exists():
            messagebox.showwarning("Archivo no encontrado",
                                   f"No se encontró:\n{path}\n\nSelecciona el TSV correcto.")
            browse_fn()
        return Path(var.get().strip())

    def run_in_thread(target, *args, **kwargs):
        btn_disable_all()
        def wrapper():
            try:
                target(*args, **kwargs)
            except SystemExit as e:
                gui_log(str(e))
            except Exception as e:
                gui_log(f"[x] Error: {e}")
            finally:
                btn_enable_all()
        threading.Thread(target=wrapper, daemon=True).start()

    def common_kwargs():
        return dict(
            input_path=var_input.get(),
            lang=var_lang.get().strip(),
            processors=var_procs.get().strip(),
            use_gpu=bool(var_gpu.get()),
            log_fn=gui_log,
        )

    def action_pretty():
        run_in_thread(run_analysis, **common_kwargs(), do_pretty=True)

    def action_tsv():
        out = Path(var_out_tsv.get() or "salida.tsv")
        run_in_thread(run_analysis, **common_kwargs(), out_tsv=out)

    def action_conllu():
        out = Path(var_out_conllu.get() or "salida.conllu")
        run_in_thread(run_analysis, **common_kwargs(), out_conllu=out)

    def action_xlsx():
        out = Path(var_out_xlsx.get() or "salida.xlsx")
        run_in_thread(run_analysis, **common_kwargs(), out_xlsx=out)

    def action_all():
        base = Path(var_base.get() or "salida")
        run_in_thread(
            run_analysis,
            **common_kwargs(),
            do_pretty=True,
            out_tsv=base.with_suffix(".tsv"),
            out_conllu=base.with_suffix(".conllu"),
            out_xlsx=base.with_suffix(".xlsx"),
            out_json=base.with_suffix(".json"),
        )

    def action_plots():
        try:
            from .modules import plots as pl
        except Exception:
            import Stanza.modules.plots as pl

        def _go():
            # asegurar TSV válido
            tsv_path = ensure_tsv_exists(var_plots_tsv, browse_plots_tsv)
            outs = pl.generate_all_plots(
                tsv_path=str(tsv_path),
                out_dir=var_plots_dir.get() or "plots",
                topk_lemmas=int(var_topk_lem.get() or 20),
                topk_deprel=(int(var_topk_dep.get()) if var_topk_dep.get() else None),
                make_wordcloud=bool(var_wc.get()),
                stopwords=None,
            )
            gui_log("[✓] Gráficos guardados:")
            for k, v in outs.items():
                gui_log(f"  - {k}: {v}")

        run_in_thread(_go)

    def action_stats():
        try:
            from .modules import stats as st
        except Exception:
            from Stanza.modules import stats as st

        def _go():
            tsv_path = ensure_tsv_exists(var_stats_tsv, browse_stats_tsv)
            pack = st.build_all_stats(
                tsv_path=str(tsv_path),
                top_lemmas=int(var_stats_top.get() or 50),
                window_cooc=int(var_stats_win.get() or 2),
            )
            out_xlsx = var_stats_out.get() or "estadisticas.xlsx"
            path = st.export_stats_to_excel(pack, out_xlsx)
            gui_log(f"[✓] Estadísticas exportadas a: {path}")

        run_in_thread(_go)

    def btn_disable_all():
        for b in all_buttons:
            b.config(state="disabled")

    def btn_enable_all():
        for b in all_buttons:
            b.config(state="normal")

    # ---- Ventana ----
    root = tk.Tk()
    root.title("Stanza – Interfaz gráfica")
    root.geometry("900x650")

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)

    # ---- Config básica ----
    frm_cfg = ttk.LabelFrame(main, text="Configuración")
    frm_cfg.pack(fill="x", pady=(0, 10))

    var_input = tk.StringVar(value=cfg_init.get("default_text", DEFAULT_INPUT))
    var_lang  = tk.StringVar(value=cfg_init.get("lang", DEFAULT_LANG))
    var_procs = tk.StringVar(value=cfg_init.get("processors", DEFAULT_PROCS))
    var_gpu   = tk.IntVar(value=1 if cfg_init.get("use_gpu", DEFAULT_GPU) else 0)

    row = 0
    ttk.Label(frm_cfg, text="Archivo de entrada:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
    ent_input = ttk.Entry(frm_cfg, textvariable=var_input, width=70)
    ent_input.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
    btn_browse = ttk.Button(frm_cfg, text="Explorar…", command=browse_input)
    btn_browse.grid(row=row, column=2, padx=6, pady=6)

    frm_cfg.columnconfigure(1, weight=1)

    # --- Idioma con combobox (editable + lista) ---
    row += 1
    ttk.Label(frm_cfg, text="Idioma:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
    LANG_VALUES = ["es","en","pt","fr","it","de","ca","gl","eu","ro"]
    cmb_lang = ttk.Combobox(frm_cfg, textvariable=var_lang, values=LANG_VALUES, width=16, state="normal")
    cmb_lang.grid(row=row, column=1, sticky="w", padx=6, pady=6)
    # (state="normal" => permite escribir y también desplegar la lista)

    row += 1
    ttk.Label(frm_cfg, text="Processors:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
    ttk.Entry(frm_cfg, textvariable=var_procs).grid(row=row, column=1, sticky="ew", padx=6, pady=6)

    row += 1
    ttk.Checkbutton(frm_cfg, text="Usar GPU", variable=var_gpu).grid(row=row, column=1, sticky="w", padx=6, pady=6)

    # ---- Salidas estándar ----
    frm_out = ttk.LabelFrame(main, text="Salidas")
    frm_out.pack(fill="x", pady=(0, 10))

    var_out_tsv = tk.StringVar(value="salida.tsv")
    var_out_conllu = tk.StringVar(value="salida.conllu")
    var_out_xlsx = tk.StringVar(value="salida.xlsx")
    var_base = tk.StringVar(value="salida")

    r = 0
    ttk.Label(frm_out, text="TSV:").grid(row=r, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_out, textvariable=var_out_tsv, width=30).grid(row=r, column=1, sticky="w", padx=6, pady=4)

    r += 1
    ttk.Label(frm_out, text="CoNLL-U:").grid(row=r, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_out, textvariable=var_out_conllu, width=30).grid(row=r, column=1, sticky="w", padx=6, pady=4)

    r += 1
    ttk.Label(frm_out, text="XLSX:").grid(row=r, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_out, textvariable=var_out_xlsx, width=30).grid(row=r, column=1, sticky="w", padx=6, pady=4)

    r += 1
    ttk.Label(frm_out, text="Base (TODO):").grid(row=r, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_out, textvariable=var_base, width=30).grid(row=r, column=1, sticky="w", padx=6, pady=4)

    # ---- Plots / Stats ----
    frm_adv = ttk.LabelFrame(main, text="Gráficos y Estadísticas")
    frm_adv.pack(fill="x", pady=(0, 10))

    var_plots_tsv = tk.StringVar(value="salida.tsv")
    var_plots_dir = tk.StringVar(value="plots")
    var_topk_lem  = tk.StringVar(value="20")
    var_topk_dep  = tk.StringVar(value="")
    var_wc        = tk.IntVar(value=1)

    var_stats_tsv = tk.StringVar(value="salida.tsv")
    var_stats_top = tk.StringVar(value="50")
    var_stats_win = tk.StringVar(value="2")
    var_stats_out = tk.StringVar(value="estadisticas.xlsx")

    rr = 0
    ttk.Label(frm_adv, text="TSV (Plots):").grid(row=rr, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_plots_tsv, width=42).grid(row=rr, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frm_adv, text="Explorar…", command=browse_plots_tsv).grid(row=rr, column=2, padx=6, pady=4)
    ttk.Label(frm_adv, text="Dir plots:").grid(row=rr, column=3, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_plots_dir, width=18).grid(row=rr, column=4, sticky="w", padx=6, pady=4)

    rr += 1
    ttk.Label(frm_adv, text="Top-K lemas:").grid(row=rr, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_topk_lem, width=8).grid(row=rr, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(frm_adv, text="Top-K deprel (vacío=todos):").grid(row=rr, column=3, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_topk_dep, width=8).grid(row=rr, column=4, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(frm_adv, text="Wordcloud", variable=var_wc).grid(row=rr, column=5, sticky="w", padx=6, pady=4)

    rr += 1
    ttk.Label(frm_adv, text="TSV (Stats):").grid(row=rr, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_stats_tsv, width=42).grid(row=rr, column=1, sticky="w", padx=6, pady=4)
    ttk.Button(frm_adv, text="Explorar…", command=browse_stats_tsv).grid(row=rr, column=2, padx=6, pady=4)
    ttk.Label(frm_adv, text="Top lemas:").grid(row=rr, column=3, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_stats_top, width=6).grid(row=rr, column=4, sticky="w", padx=6, pady=4)
    ttk.Label(frm_adv, text="Ventana cooc.:").grid(row=rr, column=5, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_stats_win, width=6).grid(row=rr, column=6, sticky="w", padx=6, pady=4)
    ttk.Label(frm_adv, text="Salida XLSX:").grid(row=rr, column=7, sticky="e", padx=6, pady=4)
    ttk.Entry(frm_adv, textvariable=var_stats_out, width=18).grid(row=rr, column=8, sticky="w", padx=6, pady=4)

    # ---- Botonera ----
    frm_btn = ttk.Frame(main)
    frm_btn.pack(fill="x", pady=(0, 10))

    btn_pretty = ttk.Button(frm_btn, text="Pretty", command=action_pretty)
    btn_tsv    = ttk.Button(frm_btn, text="TSV", command=action_tsv)
    btn_conllu = ttk.Button(frm_btn, text="CoNLL-U", command=action_conllu)
    btn_xlsx   = ttk.Button(frm_btn, text="XLSX", command=action_xlsx)
    btn_all    = ttk.Button(frm_btn, text="TODO", command=action_all)
    btn_plots  = ttk.Button(frm_btn, text="Plots", command=action_plots)
    btn_stats  = ttk.Button(frm_btn, text="Stats", command=action_stats)

    for i, b in enumerate([btn_pretty, btn_tsv, btn_conllu, btn_xlsx, btn_all, btn_plots, btn_stats]):
        b.grid(row=0, column=i, padx=6, pady=6)

    all_buttons = [btn_pretty, btn_tsv, btn_conllu, btn_xlsx, btn_all, btn_plots, btn_stats]

    # ---- Log ----
    frm_log = ttk.LabelFrame(main, text="Consola")
    frm_log.pack(fill="both", expand=True)

    txt_log = tk.Text(frm_log, height=14, wrap="word")
    txt_log.pack(fill="both", expand=True)
    txt_log.configure(state="disabled")

    # Auto-drain de logs cada 120 ms
    drain_log()

    root.mainloop()


# ===================== Main =====================

def main():
    # Import tardío del menú (con fallback)
    try:
        from .menu import menu_loop, MenuConfig, MenuChoice
    except Exception:
        from menu import menu_loop, MenuConfig, MenuChoice

    cfg_json = load_config()

    # CLI (GUI por defecto; consola solo con --cli)
    parser = argparse.ArgumentParser(description="Frontend del proyecto Stanza (GUI por defecto).")
    parser.add_argument("-i", "--input", default=cfg_json.get("default_text", DEFAULT_INPUT),
                        help="Archivo de entrada (texto plano).")
    parser.add_argument("-l", "--lang", default=cfg_json.get("lang", DEFAULT_LANG),
                        help="Idioma Stanza (ej. es, en...).")
    parser.add_argument("-p", "--processors", default=cfg_json.get("processors", DEFAULT_PROCS),
                        help="Procesadores de Stanza (coma).")
    parser.add_argument("--gpu", action="store_true", default=cfg_json.get("use_gpu", DEFAULT_GPU),
                        help="Usar GPU si está disponible.")
    parser.add_argument("--quick-tsv", action="store_true",
                        help="Modo rápido: genera salida.tsv (sin GUI).")
    parser.add_argument("--cli", action="store_true",
                        help="Usar menú de texto en lugar de la GUI.")
    args = parser.parse_args()

    # Persistir últimos argumentos
    cfg_json.update({
        "default_text": args.input,
        "lang": args.lang,
        "processors": args.processors,
        "use_gpu": bool(args.gpu),
    })
    save_config(cfg_json)

    # Modo rápido sin GUI
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

    # GUI por defecto
    if not args.cli:
        launch_gui(cfg_json)
        return

    # ---- CLI (solo si se pide --cli) ----
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
            try:
                from .modules import plots as pl
            except Exception:
                import Stanza.modules.plots as pl
            outs = pl.generate_all_plots(
                tsv_path=choice.plots_from_tsv or "salida.tsv",
                out_dir=choice.plots_out_dir or "plots",
                topk_lemmas=choice.topk_lemmas or 20,
                topk_deprel=choice.topk_deprel,
                make_wordcloud=bool(choice.wordcloud),
                stopwords=None,
            )
            print("[✓] Gráficos guardados:")
            for k, v in outs.items():
                print(f"  - {k}: {v}")

        elif choice.kind == "settings":
            save_config({
                "default_text": cfg.input_path,
                "lang": cfg.lang,
                "processors": cfg.processors,
                "use_gpu": cfg.use_gpu,
            })
            print("[✓] Configuración actualizada y guardada en config.json.")

        elif choice.kind == "stats":
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


