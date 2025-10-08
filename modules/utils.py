# -*- coding: utf-8 -*-
"""
utils.py
--------
Utilidades comunes para el proyecto:
- Manejo de rutas y archivos (lectura/escritura segura, extensiones, carpetas)
- Carga/guardado de configuración JSON
- Búsqueda de archivos relativos al script o al CWD
- Validación ligera de la cadena 'processors' de Stanza
- Pequeñas utilidades (timestamp, slugify)

Este módulo NO depende de Stanza; es agnóstico.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import re
import time
import shutil
import tempfile

# ============== Rutas / Archivos ==============

def project_root(file_: str = __file__) -> Path:
    """Retorna la carpeta raíz del módulo actual (donde vive utils.py)."""
    return Path(file_).resolve().parent

def resolve_path(path_str: str, *, base: Path | None = None) -> Path:
    """
    Resuelve una ruta probando (1) tal cual / CWD y (2) relativa a `base` (por defecto, el directorio del módulo).
    Útil cuando el usuario ejecuta desde otra carpeta.
    """
    p = Path(path_str)
    if p.exists():
        return p
    if base is None:
        base = project_root()
    p2 = base / path_str
    return p2

def ensure_parent_dir(path: Path) -> None:
    """Crea la carpeta contenedora si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)

def ensure_suffix(p: Path, suffix: str) -> Path:
    """Añade la extensión si falta."""
    return p if p.suffix.lower() == suffix.lower() else p.with_suffix(suffix)

def read_text_smart(path_str: str, *, fallback_encoding: str = "utf-8") -> str:
    """
    Lee texto intentando UTF-8; si falla y está instalado chardet/cchardet, intenta detectar encoding.
    """
    p = Path(path_str)
    if not p.exists():
        # probar relativo al módulo
        p = resolve_path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {Path(path_str).resolve()}")

    try:
        return p.read_text(encoding=fallback_encoding)
    except UnicodeDecodeError:
        # Intentar detección si existe chardet
        try:
            import chardet  # type: ignore
            raw = p.read_bytes()
            enc = chardet.detect(raw).get("encoding") or fallback_encoding
            return raw.decode(enc, errors="replace")
        except Exception:
            # última opción: latin-1
            return p.read_text(encoding="latin-1", errors="replace")

def write_text_atomic(path: Path | str, text: str, *, encoding: str = "utf-8") -> Path:
    """
    Escritura atómica: escribe primero en un archivo temporal y luego reemplaza.
    Evita archivos truncados si hay interrupciones.
    """
    path = Path(path)
    ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding) as tf:
        tf.write(text)
        tmp_name = tf.name
    shutil.move(tmp_name, path)
    return path

# ============== Configuración JSON ==============

def load_json_config(config_path: Path | str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Carga un JSON de configuración; si no existe, lo crea con defaults.
    Si está corrupto, retorna defaults y no sobreescribe automáticamente.
    """
    p = Path(config_path)
    if not p.exists():
        ensure_parent_dir(p)
        p.write_text(json.dumps(defaults, indent=4, ensure_ascii=False), encoding="utf-8")
        return dict(defaults)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # merge suave: defaults <- data
        merged = dict(defaults)
        merged.update({k: v for k, v in data.items() if k in defaults})
        return merged
    except json.JSONDecodeError:
        # mantener archivo tal cual y devolver defaults
        return dict(defaults)

def save_json_config(config_path: Path | str, data: Dict[str, Any], *, allowed_keys: set[str] | None = None) -> Path:
    """
    Guarda un dict como JSON; opcionalmente filtra por claves permitidas.
    """
    p = Path(config_path)
    ensure_parent_dir(p)
    payload = {k: v for k, v in data.items() if (allowed_keys is None or k in allowed_keys)}
    p.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    return p

# ============== Validación de 'processors' (Stanza) ==============

_ALLOWED_PROCESSORS = {
    "tokenize", "mwt", "pos", "lemma", "depparse", "ner",
    # otros posibles: "sentiment", "constituency", "depparse", "coref" (según modelos instalados)
}

def validate_processors(proc_str: str) -> Tuple[str, list[str]]:
    """
    Validación ligera de la cadena 'processors' de Stanza.
    - Elimina espacios extra.
    - Quita duplicados preservando el orden.
    - Advierte si hay procesadores desconocidos.
    - Reordena mínimamente para que 'tokenize' vaya primero y 'mwt' después si aparecen.
    Retorna: (proc_normalizado, warnings)
    """
    warnings: list[str] = []
    parts_raw = [p.strip() for p in proc_str.split(",") if p.strip()]
    seen = set()
    parts = []
    for p in parts_raw:
        if p not in seen:
            parts.append(p)
            seen.add(p)

    unknown = [p for p in parts if p not in _ALLOWED_PROCESSORS]
    if unknown:
        warnings.append(f"Procesadores no reconocidos: {', '.join(unknown)}")

    # reordenar suavemente: tokenize -> mwt -> pos -> lemma -> depparse -> ner -> (resto)
    priority = {name: i for i, name in enumerate(["tokenize", "mwt", "pos", "lemma", "depparse", "ner"])}
    parts_sorted = sorted(parts, key=lambda x: priority.get(x, 999))

    # asegurar tokenize primero si existe alguno más
    if "tokenize" in parts_sorted and parts_sorted[0] != "tokenize":
        warnings.append("Reordenado: 'tokenize' movido al inicio.")
        parts_sorted.remove("tokenize")
        parts_sorted.insert(0, "tokenize")

    normalized = ",".join(parts_sorted)
    return normalized, warnings

# ============== Pequeñas utilidades varias ==============

def timestamp_str(fmt: str = "%Y%m%d-%H%M%S") -> str:
    """Retorna un timestamp (local time) formateado para nombres de archivo."""
    return time.strftime(fmt, time.localtime())

_slug_re = re.compile(r"[^a-zA-Z0-9._-]+")

def slugify_filename(name: str, *, lower: bool = True) -> str:
    """
    Convierte una cadena en un nombre de archivo “seguro”.
    - Reemplaza espacios/caracteres raros por '-'
    - Mantiene . _ -
    """
    s = name.strip().replace(" ", "-")
    s = _slug_re.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s.lower() if lower else s

def find_default_input(default_name: str = "mi_texto.txt") -> Optional[Path]:
    """
    Intenta encontrar un archivo de texto por defecto:
    - En el CWD
    - Relativo al módulo
    Retorna Path o None.
    """
    p1 = Path(default_name)
    if p1.exists():
        return p1
    p2 = project_root() / default_name
    if p2.exists():
        return p2
    return None

def require_package(pkg_name: str, install_hint: str | None = None) -> None:
    """
    Verifica que un paquete esté disponible. Lanza RuntimeError con hint si no lo está.
    Útil en puntos de extensión (e.g., al generar Excel, plots, etc.).
    """
    try:
        __import__(pkg_name)
    except Exception as e:
        hint = f"\nInstala con: pip install {install_hint or pkg_name}"
        raise RuntimeError(f"Falta el paquete requerido: {pkg_name}.{hint}") from e


__all__ = [
    # rutas/archivos
    "project_root", "resolve_path", "ensure_parent_dir",
    "ensure_suffix", "read_text_smart", "write_text_atomic",
    # config
    "load_json_config", "save_json_config",
    # processors
    "validate_processors",
    # misceláneo
    "timestamp_str", "slugify_filename", "find_default_input", "require_package",
]
