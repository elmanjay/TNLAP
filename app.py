import json
import math
import random
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# PAGE GEOMETRY
# Neues Format: keine "canvas"-Schlüssel mehr.
# Geometrie ist in Grid-Einheiten (x: 0-6, y: 0-10).
# Wir leiten die Canvas-Größe aus den tatsächlichen Geometrien ab.
# ============================================================
DEFAULT_GRID_W = 6
DEFAULT_GRID_H = 10


# -----------------------------
# Instance helpers
# -----------------------------
def get_page_layouts(instance: dict, page_id: int) -> List[int]:
    """layouts_pages[page_id] = [layout_ids]  — unverändert"""
    layouts = instance.get("layouts_pages", {}).get(str(page_id), [])
    return [int(x) for x in layouts]


def get_layout_boxes(instance: dict, layout_id: int) -> List[int]:
    """box_layouts[layout_id] = [box_ids]  — unverändert"""
    boxes = instance.get("box_layouts", {}).get(str(layout_id), [])
    return [int(b) for b in boxes]


def shells_for_layout_box(instance: dict, layout_id: int, box_id: int) -> List[int]:
    """NEU: shells_layout_box  (früher: hull_layout_box)"""
    return [
        int(h)
        for h in instance.get("shells_layout_box", {})
        .get(str(layout_id), {})
        .get(str(box_id), [])
    ]


def shell_params(instance: dict, shell_id: int) -> Dict[str, float]:
    """NEU: shell_params  (früher: hull_params)"""
    return instance.get("shell_params", {}).get(str(shell_id), {"min": 0, "max": 0})


def article_len(instance: dict, art_id: int) -> int:
    return int(instance.get("article_length", {}).get(str(art_id), 0))


def article_prio(instance: dict, art_id: int) -> str:
    return str(instance.get("article_priority", {}).get(str(art_id), "?"))


def article_sections(instance: dict, art_id: int) -> List[int]:
    """NEU: article_sections[article_id] = [section_ids]"""
    return [int(s) for s in instance.get("article_sections", {}).get(str(art_id), [])]


def canvas_wh(instance: dict) -> Tuple[float, float]:
    """
    NEU: Kein 'canvas'-Schlüssel mehr.
    Wir ermitteln die tatsächliche Grid-Ausdehnung aus geometry_layout_box
    oder fallen auf DEFAULT_GRID_W / DEFAULT_GRID_H zurück.
    """
    geom_all = instance.get("geometry_layout_box", {})
    max_x, max_y = 0.0, 0.0
    for boxes in geom_all.values():
        for g in boxes.values():
            max_x = max(max_x, float(g.get("x", 0)) + float(g.get("w", 0)))
            max_y = max(max_y, float(g.get("y", 0)) + float(g.get("h", 0)))
    W = max_x if max_x > 0 else float(DEFAULT_GRID_W)
    H = max_y if max_y > 0 else float(DEFAULT_GRID_H)
    return W, H


def get_box_geometry(instance: dict, layout_id: int, box_id: int) -> Optional[Dict[str, float]]:
    """
    NEU: geometry_layout_box[layout][box] hat jetzt 'character' statt 'area'.
    Wir liefern beide Keys (area = character) für Abwärtskompatibilität im UI.
    """
    g = (
        instance.get("geometry_layout_box", {})
        .get(str(layout_id), {})
        .get(str(box_id))
    )
    if not g:
        return None
    w = float(g.get("w", 0.0))
    h = float(g.get("h", 0.0))
    # 'character' ersetzt 'area' im neuen Format
    character = float(g.get("character", g.get("area", w * h)))
    return {
        "x": float(g.get("x", 0.0)),
        "y": float(g.get("y", 0.0)),
        "w": w,
        "h": h,
        "area": w * h,          # geometrische Fläche (Grid-Einheiten²)
        "character": character, # Zeichenkapazität der Box
    }


# -----------------------------
# NEU: shell->articles Mapping
# shells_article[article_id] = [shell_ids]
# Wir invertieren: shell_id -> [article_ids]
# -----------------------------
def build_shell_to_articles(instance: dict) -> Dict[int, List[int]]:
    """
    NEU: shells_article  (früher: hull_article – gleiche Richtung, neuer Name)
    shells_article[article_id] = [shell_ids]
    Invertieren zu shell_id -> [article_ids].
    """
    valid_articles = set(int(a) for a in instance.get("article", []))
    raw = instance.get("shells_article", {})

    shell_to_articles: Dict[int, List[int]] = {}

    for ak, sv in raw.items():
        try:
            a = int(ak)
        except Exception:
            continue
        if a not in valid_articles:
            continue
        if not isinstance(sv, list):
            continue
        for s in sv:
            try:
                si = int(s)
            except Exception:
                continue
            shell_to_articles.setdefault(si, []).append(a)

    for s in list(shell_to_articles.keys()):
        shell_to_articles[s] = sorted(set(shell_to_articles[s]))

    return shell_to_articles


def compatible_articles_for_shell(instance: dict, shell_id: int) -> List[int]:
    """Gibt alle Artikel zurück, die mit diesem Shell kompatibel sind."""
    if "_shell_to_articles" not in st.session_state:
        st.session_state["_shell_to_articles"] = build_shell_to_articles(instance)

    valid_articles = set(int(a) for a in instance.get("article", []))
    m = st.session_state["_shell_to_articles"]
    return [int(a) for a in m.get(int(shell_id), []) if int(a) in valid_articles]


# -----------------------------
# Sections-Hilfsfunktionen (korrekte Datenstruktur)
# -----------------------------
def get_sections_for_page(instance: dict, page_id: int) -> List[int]:
    """sections_page[page_id] = [section_ids] — welche Sections sind auf dieser Seite verfügbar"""
    return [int(s) for s in instance.get("sections_page", {}).get(str(page_id), [])]


def get_articles_for_section(instance: dict, section_id: int) -> List[int]:
    """article_sections[section_id] = [article_ids] — welche Artikel gehören zu dieser Section"""
    return [int(a) for a in instance.get("article_sections", {}).get(str(section_id), [])]


def get_pages_for_article(instance: dict, art_id: int) -> List[int]:
    """Abgeleitet: Artikel ist auf Seite X verfügbar wenn seine Section auf X aktiv ist"""
    pages = []
    for page_id in instance.get("pages", []):
        for sec in get_sections_for_page(instance, int(page_id)):
            if int(art_id) in get_articles_for_section(instance, sec):
                pages.append(int(page_id))
    return sorted(set(pages))


def get_section_for_article_on_page(instance: dict, art_id: int, page_id: int) -> Optional[int]:
    """Welche Section verbindet diesen Artikel mit dieser Seite?"""
    for sec in get_sections_for_page(instance, page_id):
        if int(art_id) in get_articles_for_section(instance, sec):
            return sec
    return None


# -----------------------------
# Build rectangles dataframe for Altair (REAL GEOMETRY)
# -----------------------------
def layout_rects_df(instance: dict, layout_id: int) -> pd.DataFrame:
    W, H = canvas_wh(instance)

    boxes = get_layout_boxes(instance, layout_id)
    if not boxes:
        return pd.DataFrame(
            columns=["layout", "box", "x0", "x1", "y0", "y1", "w", "h", "area", "character", "shells", "num_shells"]
        )

    rows = []
    for b in boxes:
        geom = get_box_geometry(instance, layout_id, b)
        if geom is None:
            continue

        x = geom["x"]
        y = geom["y"]
        w = geom["w"]
        h = geom["h"]

        hs = shells_for_layout_box(instance, layout_id, b)
        rows.append(
            {
                "layout": int(layout_id),
                "box": int(b),
                "x0": float(x),
                "x1": float(x + w),
                "y0": float(y),
                "y1": float(y + h),
                "w": float(w),
                "h": float(h),
                "area": float(geom["area"]),
                "character": float(geom["character"]),
                "shells": ", ".join(map(str, hs)),
                "num_shells": int(len(hs)),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["x0", "x1", "y0", "y1"]).reset_index(drop=True)
    return df


# -----------------------------
# Newspaper layout helpers
# -----------------------------
# Margin in Grid-Einheiten (wird zu den 6×10 Inhaltskoordinaten addiert).
# Der Satzspiegel liegt bei (MX, MY) bis (MX+W, MY+H).
# Das Gesamtdokument hat die Ausdehnung (0..W+2*MX) × (0..H+2*MY).
MARGIN_X = 0.35   # links & rechts
MARGIN_Y = 0.45   # oben & unten
N_COLS = 6        # Spaltenanzahl


def _no_axis():
    return alt.Axis(labels=False, ticks=False, domain=False, grid=False, title=None)


def _newspaper_layers(instance: dict, width_px: int):
    """
    Gibt (layers_list, full_W, full_H, height_px) zurück.
    layers_list enthält Altair-Charts die mit + kombiniert werden können:
      - Papier-Hintergrund (cream)
      - Margin-Schatten (dunklere Randbereiche außerhalb des Satzspiegels)
      - Satzspiegel-Fläche (weiß)
      - 6 Spaltengitter-Linien (gestrichelt, sehr dezent)
      - Äußerer Seitenrand (schwarz, dünn)
      - Satzspiegel-Rahmen (schwarz, etwas stärker)
    """
    W, H = canvas_wh(instance)
    MX, MY = MARGIN_X, MARGIN_Y
    FW = W + 2 * MX   # full document width in grid units
    FH = H + 2 * MY   # full document height
    height_px = int(width_px * (FH / FW))

    NO_AXIS = _no_axis()
    xs = alt.Scale(domain=[0, FW], nice=False)
    ys = alt.Scale(domain=[0, FH], nice=False, reverse=True)  # y=0 oben

    def xenc(field):
        return alt.X(f"{field}:Q", axis=NO_AXIS, scale=xs)
    def yenc(field):
        return alt.Y(f"{field}:Q", axis=NO_AXIS, scale=ys)

    # --- Papier-Hintergrund (cream) ---
    bg_df = pd.DataFrame([{"x0": 0, "x1": FW, "y0": 0, "y1": FH}])
    bg = (
        alt.Chart(bg_df)
        .mark_rect(fill="#fdf6e3", stroke=None)
        .encode(xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q"))
        .properties(width=width_px, height=height_px)
    )

    # --- Margin-Fläche (4 Streifen = Schatten außerhalb Satzspiegel) ---
    # links, rechts, oben, unten
    margin_rects = [
        {"x0": 0,       "x1": MX,      "y0": 0,  "y1": FH},   # links
        {"x0": MX + W,  "x1": FW,      "y0": 0,  "y1": FH},   # rechts
        {"x0": MX,      "x1": MX + W,  "y0": 0,  "y1": MY},   # oben
        {"x0": MX,      "x1": MX + W,  "y0": MY + H, "y1": FH},  # unten
    ]
    margin_df = pd.DataFrame(margin_rects)
    margin_layer = (
        alt.Chart(margin_df)
        .mark_rect(fill="#e8dfc8", stroke=None)
        .encode(xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q"))
        .properties(width=width_px, height=height_px)
    )

    # --- Satzspiegel-Fläche (weiß) ---
    ss_df = pd.DataFrame([{"x0": MX, "x1": MX + W, "y0": MY, "y1": MY + H}])
    ss_fill = (
        alt.Chart(ss_df)
        .mark_rect(fill="#ffffff", stroke=None)
        .encode(xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q"))
        .properties(width=width_px, height=height_px)
    )

    # --- Spaltengitter (6 Spalten, gestrichelte Linien an den Grenzen) ---
    col_w = W / N_COLS
    col_lines = []
    for i in range(1, N_COLS):
        cx = MX + i * col_w
        col_lines.append({"x0": cx, "x1": cx, "y0": MY, "y1": MY + H})
    col_df = pd.DataFrame(col_lines)
    col_grid = (
        alt.Chart(col_df)
        .mark_rule(stroke="#b0a090", strokeWidth=0.6, strokeDash=[3, 4])
        .encode(
            x=alt.X("x0:Q", scale=xs),
            y=alt.Y("y0:Q", scale=ys),
            y2="y1:Q",
        )
        .properties(width=width_px, height=height_px)
    )

    # --- Äußerer Seitenrand ---
    page_border_df = pd.DataFrame([{"x0": 0, "x1": FW, "y0": 0, "y1": FH}])
    page_border = (
        alt.Chart(page_border_df)
        .mark_rect(fillOpacity=0, stroke="#888888", strokeWidth=0.8)
        .encode(xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q"))
        .properties(width=width_px, height=height_px)
    )

    # --- Satzspiegel-Rahmen ---
    ss_border = (
        alt.Chart(ss_df)
        .mark_rect(fillOpacity=0, stroke="#333333", strokeWidth=1.2)
        .encode(xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q"))
        .properties(width=width_px, height=height_px)
    )

    return [bg, margin_layer, ss_fill, col_grid, page_border, ss_border], FW, FH, height_px


def _shift_df(df: pd.DataFrame) -> pd.DataFrame:
    """Verschiebt Box-Koordinaten um den Margin-Offset (Satzspiegel-Offset)."""
    if df.empty:
        return df
    out = df.copy()
    out["x0"] = out["x0"] + MARGIN_X
    out["x1"] = out["x1"] + MARGIN_X
    out["y0"] = out["y0"] + MARGIN_Y
    out["y1"] = out["y1"] + MARGIN_Y
    return out


def preview_chart(df: pd.DataFrame, instance: dict, width_px=180, active=False):
    if df is None or df.empty:
        return None

    base_layers, FW, FH, height_px = _newspaper_layers(instance, width_px)
    NO_AXIS = _no_axis()
    xs = alt.Scale(domain=[0, FW], nice=False)
    ys = alt.Scale(domain=[0, FH], nice=False, reverse=True)  # y=0 oben

    sdf = _shift_df(df)

    boxes = (
        alt.Chart(sdf)
        .mark_rect(stroke="#555555", strokeWidth=0.7)
        .encode(
            x=alt.X("x0:Q", axis=NO_AXIS, scale=xs),
            x2="x1:Q",
            y=alt.Y("y0:Q", axis=NO_AXIS, scale=ys),
            y2="y1:Q",
            color=alt.value("#bfdbfe" if active else "#e2e8f0"),
            tooltip=[
                alt.Tooltip("box:N", title="Box"),
                alt.Tooltip("num_shells:Q", title="#Shells"),
            ],
        )
        .properties(width=width_px, height=height_px)
    )

    chart = alt.layer(*base_layers, boxes)
    return chart.configure_view(stroke=None)


def box_chart(df: pd.DataFrame, instance: dict, width_px=460):
    """
    Detailansicht eines Layouts mit:
      - Spalten-Header-Balken oben (Sp. 1–6, jede Spalte beschriftet + Trennlinien)
      - Zeilen-Achse links (0, 10, 20 … 100)
      - Satzspiegel-Zeitungsoptik (Margins, Spaltengitter)
      - Box-Rechtecke mit zentrierter Nummer und Kurzinfo
    """
    if df is None or df.empty:
        return None

    W, H = canvas_wh(instance)          # 6, 10  (Grid-Einheiten)
    ROW_SCALE = 100 / H                  # 10 → jede Grid-Einheit = 10 Zeilen
    COL_SCALE = N_COLS / W               # 1  → jede Grid-Einheit = 1 Spalte

    MX, MY = MARGIN_X, MARGIN_Y

    # ---- Randbreiten für Achsenbeschriftungen (in Grid-Einheiten) ----
    LEFT_AXIS_W  = 0.55   # Platz für Zeilennummern links
    TOP_HEADER_H = 0.55   # Platz für Spalten-Header oben

    # Vollständiger Koordinatenraum inkl. Margin + Achsplatz
    FW = LEFT_AXIS_W + W + 2 * MX
    FH = TOP_HEADER_H + H + 2 * MY
    height_px = int(width_px * (FH / FW))

    NO_AXIS = _no_axis()
    # x: 0 = linker Rand, LEFT_AXIS_W = wo Seite beginnt
    # y: 0 = oberer Rand (reverse=True), TOP_HEADER_H = wo Seite beginnt
    xs = alt.Scale(domain=[0, FW], nice=False)
    ys = alt.Scale(domain=[0, FH], nice=False, reverse=True)

    # Offset: alles was vorher bei (0,0) startete, startet jetzt bei (LEFT_AXIS_W, TOP_HEADER_H)
    OX = LEFT_AXIS_W + MX   # x-offset für Inhalt: Achsplatz + linker Margin
    OY = TOP_HEADER_H + MY  # y-offset für Inhalt: Header-Platz + oberer Margin

    def xenc(f): return alt.X(f"{f}:Q", axis=NO_AXIS, scale=xs)
    def yenc(f): return alt.Y(f"{f}:Q", axis=NO_AXIS, scale=ys)
    def props(): return dict(width=width_px, height=height_px)

    # ----------------------------------------------------------------
    # 1. Papier-Hintergrund (gesamt)
    # ----------------------------------------------------------------
    bg_df = pd.DataFrame([{"x0": 0, "x1": FW, "y0": 0, "y1": FH}])
    bg = alt.Chart(bg_df).mark_rect(fill="#fdf6e3", stroke=None).encode(
        xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q")
    ).properties(**props())

    # ----------------------------------------------------------------
    # 2. Margin-Bereiche (außerhalb Satzspiegel, dezent dunkler)
    # ----------------------------------------------------------------
    sx0 = LEFT_AXIS_W + MX
    sx1 = LEFT_AXIS_W + MX + W
    sy0 = TOP_HEADER_H + MY
    sy1 = TOP_HEADER_H + MY + H

    margin_rects = [
        {"x0": LEFT_AXIS_W,  "x1": sx0, "y0": TOP_HEADER_H, "y1": FH},   # links
        {"x0": sx1,          "x1": FW,  "y0": TOP_HEADER_H, "y1": FH},   # rechts
        {"x0": sx0,          "x1": sx1, "y0": TOP_HEADER_H, "y1": sy0},  # oben
        {"x0": sx0,          "x1": sx1, "y0": sy1,          "y1": FH},   # unten
    ]
    margin_df = pd.DataFrame(margin_rects)
    margin_layer = alt.Chart(margin_df).mark_rect(fill="#e8dfc8", stroke=None).encode(
        xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q")
    ).properties(**props())

    # ----------------------------------------------------------------
    # 3. Satzspiegel (weiß)
    # ----------------------------------------------------------------
    ss_df = pd.DataFrame([{"x0": sx0, "x1": sx1, "y0": sy0, "y1": sy1}])
    ss_fill = alt.Chart(ss_df).mark_rect(fill="#ffffff", stroke=None).encode(
        xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q")
    ).properties(**props())

    # ----------------------------------------------------------------
    # 4. Spaltengitter im Satzspiegel
    # ----------------------------------------------------------------
    col_unit = W / N_COLS
    col_lines = [
        {"x0": sx0 + i * col_unit, "x1": sx0 + i * col_unit, "y0": sy0, "y1": sy1}
        for i in range(1, N_COLS)
    ]
    col_grid = alt.Chart(pd.DataFrame(col_lines)).mark_rule(
        stroke="#b0a090", strokeWidth=0.6, strokeDash=[3, 4]
    ).encode(
        x=alt.X("x0:Q", scale=xs), y=alt.Y("y0:Q", scale=ys), y2="y1:Q"
    ).properties(**props())

    # ----------------------------------------------------------------
    # 5. Satzspiegel-Rahmen + Seitenrand
    # ----------------------------------------------------------------
    page_border = alt.Chart(bg_df).mark_rect(
        fillOpacity=0, stroke="#888888", strokeWidth=0.8
    ).encode(xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q")).properties(**props())

    ss_border = alt.Chart(ss_df).mark_rect(
        fillOpacity=0, stroke="#333333", strokeWidth=1.2
    ).encode(xenc("x0"), alt.X2("x1:Q"), yenc("y0"), alt.Y2("y1:Q")).properties(**props())

    # ----------------------------------------------------------------
    # 6. SPALTEN-HEADER OBEN
    # Für jede Spalte: gefärbter Balken + Beschriftung "Sp. N"
    # ----------------------------------------------------------------
    HEADER_FILL   = "#334155"   # dunkles Schieferblau
    HEADER_STROKE = "#1e293b"
    HEADER_TEXT   = "#f8fafc"
    HEADER_TOP    = TOP_HEADER_H * 0.08    # kleiner Luft-Abstand zum Rand
    HEADER_BOT    = TOP_HEADER_H * 0.92

    col_header_rects = []
    col_header_labels = []
    for i in range(N_COLS):
        cx0 = sx0 + i * col_unit + 0.02
        cx1 = sx0 + (i + 1) * col_unit - 0.02
        col_header_rects.append({
            "x0": cx0, "x1": cx1,
            "y0": HEADER_TOP, "y1": HEADER_BOT,
            "col": i + 1,
        })
        col_header_labels.append({
            "lx": (cx0 + cx1) / 2,
            "ly": (HEADER_TOP + HEADER_BOT) / 2,
            "label": f"Sp. {i + 1}",
        })

    col_hdr_df  = pd.DataFrame(col_header_rects)
    col_lbl_df  = pd.DataFrame(col_header_labels)

    col_hdr_rects = alt.Chart(col_hdr_df).mark_rect(
        fill=HEADER_FILL, stroke=HEADER_STROKE, strokeWidth=0.8, cornerRadius=2
    ).encode(
        x=alt.X("x0:Q", axis=NO_AXIS, scale=xs),
        x2="x1:Q",
        y=alt.Y("y0:Q", axis=NO_AXIS, scale=ys),
        y2="y1:Q",
        tooltip=[alt.Tooltip("col:Q", title="Spalte")],
    ).properties(**props())

    col_hdr_text = alt.Chart(col_lbl_df).mark_text(
        fontSize=10, fontWeight="bold", color=HEADER_TEXT,
        baseline="middle", align="center"
    ).encode(
        x=alt.X("lx:Q", axis=NO_AXIS, scale=xs),
        y=alt.Y("ly:Q", axis=NO_AXIS, scale=ys),
        text=alt.Text("label:N"),
    ).properties(**props())

    # Trennlinie unter dem Header (Satzspiegel-Oberkante)
    hdr_line_df = pd.DataFrame([{"x0": sx0, "x1": sx1, "y": sy0}])
    hdr_line = alt.Chart(hdr_line_df).mark_rule(
        stroke="#334155", strokeWidth=1.5
    ).encode(
        x=alt.X("x0:Q", scale=xs), x2="x1:Q",
        y=alt.Y("y:Q", scale=ys),
    ).properties(**props())

    # ----------------------------------------------------------------
    # 7. ZEILEN-ACHSE LINKS
    # Ticks bei 0, 10, 20 … 100 Zeilen
    # ----------------------------------------------------------------
    ROW_TICKS = list(range(0, 101, 10))   # 0, 10, 20 … 100
    TICK_X1   = LEFT_AXIS_W - 0.04        # rechte Kante des Tick-Striches
    TICK_X0   = LEFT_AXIS_W - 0.12        # linke Kante (Länge des Striches)
    LABEL_X   = LEFT_AXIS_W - 0.14        # Textposition

    row_ticks_data = []
    row_labels_data = []
    for row_num in ROW_TICKS:
        gy = OY + row_num / ROW_SCALE   # grid-y Position
        row_ticks_data.append({"x0": TICK_X0, "x1": TICK_X1, "y": gy})
        row_labels_data.append({"lx": LABEL_X, "ly": gy, "label": str(row_num)})

    tick_df  = pd.DataFrame(row_ticks_data)
    label_df = pd.DataFrame(row_labels_data)

    row_ticks = alt.Chart(tick_df).mark_rule(
        stroke="#64748b", strokeWidth=1.0
    ).encode(
        x=alt.X("x0:Q", scale=xs), x2="x1:Q",
        y=alt.Y("y:Q", scale=ys),
    ).properties(**props())

    row_labels = alt.Chart(label_df).mark_text(
        fontSize=9, color="#475569", baseline="middle", align="right"
    ).encode(
        x=alt.X("lx:Q", axis=NO_AXIS, scale=xs),
        y=alt.Y("ly:Q", axis=NO_AXIS, scale=ys),
        text=alt.Text("label:N"),
    ).properties(**props())

    # Achslinie (vertikale Linie am linken Rand des Satzspiegels)
    axis_line_df = pd.DataFrame([{"x": sx0 - MX * 0.5, "y0": sy0, "y1": sy1}])
    axis_line = alt.Chart(axis_line_df).mark_rule(
        stroke="#94a3b8", strokeWidth=0.8
    ).encode(
        x=alt.X("x:Q", scale=xs),
        y=alt.Y("y0:Q", scale=ys), y2="y1:Q",
    ).properties(**props())

    # Dezente Hilfslinien über die Seitenbreite (alle 10 Zeilen)
    hgrid_data = [
        {"x0": sx0, "x1": sx1, "y": OY + r / ROW_SCALE}
        for r in ROW_TICKS[1:-1]   # ohne 0 und 100 (die hat der Rahmen)
    ]
    hgrid = alt.Chart(pd.DataFrame(hgrid_data)).mark_rule(
        stroke="#d1cbc0", strokeWidth=0.4, strokeDash=[2, 3]
    ).encode(
        x=alt.X("x0:Q", scale=xs), x2="x1:Q",
        y=alt.Y("y:Q", scale=ys),
    ).properties(**props())

    # ----------------------------------------------------------------
    # 8. BOX-RECHTECKE
    # ----------------------------------------------------------------
    sdf = df.copy()
    # Koordinaten in Spalten/Zeilen für Tooltips
    sdf["col_start"] = (sdf["x0"] * COL_SCALE).round().astype(int) + 1
    sdf["col_end"]   = (sdf["x1"] * COL_SCALE).round().astype(int)
    sdf["row_start"] = (sdf["y0"] * ROW_SCALE).round().astype(int) + 1
    sdf["row_end"]   = (sdf["y1"] * ROW_SCALE).round().astype(int)
    sdf["n_cols"]    = (sdf["w"]  * COL_SCALE).round().astype(int)
    sdf["n_rows"]    = (sdf["h"]  * ROW_SCALE).round().astype(int)
    sdf["col_label"] = sdf["n_cols"].astype(str) + " Sp. / " + sdf["n_rows"].astype(str) + " Z."

    # Auf Display-Koordinaten verschieben
    sdf["px0"] = sdf["x0"] + OX
    sdf["px1"] = sdf["x1"] + OX
    sdf["py0"] = sdf["y0"] + OY
    sdf["py1"] = sdf["y1"] + OY
    sdf["plx"] = (sdf["px0"] + sdf["px1"]) / 2
    sdf["ply"] = (sdf["py0"] + sdf["py1"]) / 2
    sdf["psy"] = sdf["py0"] + 0.10   # sublabel y (knapp unter Oberkante)

    box_rects = alt.Chart(sdf).mark_rect(
        stroke="#1e293b", strokeWidth=1.4
    ).encode(
        x=alt.X("px0:Q", axis=NO_AXIS, scale=xs),
        x2="px1:Q",
        y=alt.Y("py0:Q", axis=NO_AXIS, scale=ys),
        y2="py1:Q",
        color=alt.value("#dbeafe"),
        tooltip=[
            alt.Tooltip("box:N",       title="Box"),
            alt.Tooltip("col_start:Q", title="Sp. von"),
            alt.Tooltip("col_end:Q",   title="Sp. bis"),
            alt.Tooltip("n_cols:Q",    title="Breite (Sp.)"),
            alt.Tooltip("row_start:Q", title="Zeile von"),
            alt.Tooltip("row_end:Q",   title="Zeile bis"),
            alt.Tooltip("n_rows:Q",    title="Höhe (Zeilen)"),
            alt.Tooltip("num_shells:Q",title="#Shells"),
            alt.Tooltip("character:Q", title="capacity (chars)"),
        ],
    ).properties(**props())

    box_num = alt.Chart(sdf).mark_text(
        fontSize=14, fontWeight="bold", color="#1e3a5f",
        baseline="middle", align="center"
    ).encode(
        x=alt.X("plx:Q", axis=NO_AXIS, scale=xs),
        y=alt.Y("ply:Q", axis=NO_AXIS, scale=ys),
        text=alt.Text("box:N"),
    ).properties(**props())

    box_sub = alt.Chart(sdf).mark_text(
        fontSize=8, color="#475569", baseline="top", align="center"
    ).encode(
        x=alt.X("plx:Q", axis=NO_AXIS, scale=xs),
        y=alt.Y("psy:Q", axis=NO_AXIS, scale=ys),
        text=alt.Text("col_label:N"),
    ).properties(**props())

    chart = alt.layer(
        bg, margin_layer, ss_fill,
        hgrid, col_grid,
        page_border, ss_border,
        col_hdr_rects, col_hdr_text, hdr_line,
        axis_line, row_ticks, row_labels,
        box_rects, box_sub, box_num,
    )
    return chart.configure_view(stroke=None)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Newspaper Instance Explorer", layout="wide")

st.title("🗞️ Newspaper Instance Explorer")
st.caption(
    "Neues Instanzformat: Seiten → Layouts → Boxen → **Shells** → kompatible Artikel · "
    "inkl. Sections-Übersicht und Under-/Overfill-Filter"
)

with st.sidebar:
    st.header("📦 Instanz laden")
    up = st.file_uploader("JSON-Datei (neue Instanz)", type=["json"])
    if up is None:
        st.info("Lade eine JSON-Datei hoch.")
        st.stop()

    instance = json.load(up)
    st.success("Instanz geladen ✅")

    # Cache-Reset bei neuem File
    inst_sig = (getattr(up, "name", None), getattr(up, "size", None))
    if st.session_state.get("_instance_sig") != inst_sig:
        st.session_state["_instance_sig"] = inst_sig
        st.session_state.pop("_shell_to_articles", None)

    W, H = canvas_wh(instance)
    st.caption(f"Grid-Ausdehnung: **{W:.0f} × {H:.0f}** Einheiten")

    # Statistiken
    n_pages    = len(instance.get("pages", []))
    n_layouts  = len(instance.get("layouts", []))
    n_articles = len(instance.get("article", []))
    n_shells   = len(instance.get("shells", []))
    n_sections = len(instance.get("sections", []))
    st.markdown(
        f"**Seiten:** {n_pages} · **Layouts:** {n_layouts} · "
        f"**Artikel:** {n_articles} · **Shells:** {n_shells} · **Sections:** {n_sections}"
    )

    pages = [int(p) for p in instance.get("pages", [])]
    if not pages:
        st.error("In der JSON fehlen `pages`.")
        st.stop()

    st.divider()
    page_id = st.selectbox("Seite wählen", pages, index=0)

    layouts = get_page_layouts(instance, page_id)
    if not layouts:
        st.warning(f"Keine Layouts für Seite {page_id} gefunden.")
        st.stop()

    st.write(f"Layouts auf Seite {page_id}: **{len(layouts)}**")

# ---- Session state ----
if "chosen_layout" not in st.session_state:
    st.session_state["chosen_layout"] = int(layouts[0])
if "chosen_box" not in st.session_state:
    st.session_state["chosen_box"] = None

# ============================================================
# Tab-Navigation: Explorer | Sections
# ============================================================
tab_explorer, tab_sections, tab_solution = st.tabs(["📐 Layout Explorer", "📋 Sections", "✅ Solution"])

# ============================================================
# TAB 1: Layout Explorer (wie bisher, aber mit Shells)
# ============================================================
with tab_explorer:
    st.subheader(f"Seite {page_id}: Layout-Vorschau")

    cols = 4
    nrows = math.ceil(len(layouts) / cols)

    for r in range(nrows):
        ccols = st.columns(cols, gap="medium")
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(layouts):
                continue
            lid = int(layouts[idx])

            with ccols[c]:
                is_active = (st.session_state["chosen_layout"] == lid)
                btn_label = f"✅ Layout {lid}" if is_active else f"Layout {lid}"
                if st.button(btn_label, key=f"pick_layout_{page_id}_{lid}", use_container_width=True):
                    st.session_state["chosen_layout"] = lid
                    st.session_state["chosen_box"] = None

                df_prev = layout_rects_df(instance, lid)
                prev_chart = preview_chart(df_prev, instance, width_px=180, active=is_active)
                if prev_chart is None:
                    st.warning("Leeres Layout.")
                else:
                    st.altair_chart(prev_chart, use_container_width=False)

    st.divider()

    chosen_layout = int(st.session_state["chosen_layout"])
    st.subheader(f"Layout {chosen_layout}: Boxen untersuchen")

    df = layout_rects_df(instance, chosen_layout)
    main_chart = box_chart(df, instance, width_px=460)
    if main_chart is None:
        st.error("Keine Box-Daten für dieses Layout.")
        st.stop()
    st.altair_chart(main_chart, use_container_width=False)

    # Tabelle mit Spalten/Zeilen statt rohen Grid-Einheiten
    _, H_grid = canvas_wh(instance)
    ROW_SCALE_ui = 100 / H_grid
    tbl = df[["box", "num_shells", "w", "h", "character"]].copy()
    tbl["Sp. von"] = (df["x0"] * 1).round().astype(int) + 1
    tbl["Sp. bis"] = (df["x1"] * 1).round().astype(int)
    tbl["Zeile von"] = (df["y0"] * ROW_SCALE_ui).round().astype(int) + 1
    tbl["Zeile bis"] = (df["y1"] * ROW_SCALE_ui).round().astype(int)
    tbl["Breite (Sp.)"] = (df["w"]).round().astype(int)
    tbl["Höhe (Z.)"] = (df["h"] * ROW_SCALE_ui).round().astype(int)
    box_table = tbl[["box", "Sp. von", "Sp. bis", "Breite (Sp.)", "Zeile von", "Zeile bis", "Höhe (Z.)", "num_shells", "character"]].sort_values("box").reset_index(drop=True)
    box_table = box_table.rename(columns={"character": "capacity (chars)", "num_shells": "#shells"})
    st.caption("⬇️ Box-Auswahl")
    st.dataframe(box_table, use_container_width=True, hide_index=True)

    box_list = box_table["box"].tolist()
    if not box_list:
        st.warning("Keine Boxen in diesem Layout.")
        st.stop()

    default_idx = 0
    if st.session_state.get("chosen_box") in box_list:
        default_idx = box_list.index(st.session_state["chosen_box"])

    chosen_box = st.selectbox("Box auswählen", box_list, index=default_idx, key="chosen_box_selectbox")
    st.session_state["chosen_box"] = int(chosen_box)

    # Box / Shell explorer
    colA, colB = st.columns([1, 1.3], gap="large")

    with colA:
        st.markdown("### 🔳 Box-Details")
        st.write(f"**Box:** {chosen_box}")

        geom = get_box_geometry(instance, chosen_layout, chosen_box)
        if geom:
            st.write(
                f"**Geometrie (Grid):** x={geom['x']:.0f}, y={geom['y']:.0f}, "
                f"w={geom['w']:.0f}, h={geom['h']:.0f}"
            )
            st.write(f"**Zeichenkapazität:** {int(geom['character'])} chars")

        hs = shells_for_layout_box(instance, chosen_layout, chosen_box)
        st.write(f"**Shells in dieser Box:** {hs if hs else '—'}")

        if hs:
            # Filterwerte aus session_state lesen (können schon gesetzt sein)
            filter_on  = st.session_state.get("filter_on", False)
            under_thr  = st.session_state.get("under_thr", 15.0)
            over_thr   = st.session_state.get("over_thr",  20.0)

            allowed_arts = set()
            for sec in get_sections_for_page(instance, page_id):
                allowed_arts.update(get_articles_for_section(instance, sec))

            shell_rows = []
            for h in hs:
                sp   = shell_params(instance, h)
                hmin = int(sp.get("min", 0))
                hmax = int(sp.get("max", 0))

                arts_h = [
                    a for a in compatible_articles_for_shell(instance, h)
                    if a in allowed_arts
                ]

                if filter_on:
                    def _passes(a):
                        L = article_len(instance, a)
                        u = 100.0 * (hmin - L) / hmin if L < hmin and hmin > 0 else 0.0
                        o = 100.0 * (L - hmax) / hmax if L > hmax and hmax > 0 else 0.0
                        return u <= under_thr and o <= over_thr
                    arts_h = [a for a in arts_h if _passes(a)]

                shell_rows.append({
                    "shell":     h,
                    "min":       hmin,
                    "max":       hmax,
                    "#articles": len(arts_h),
                })
            shell_df = pd.DataFrame(shell_rows).sort_values("shell")
            st.dataframe(shell_df, use_container_width=True, hide_index=True)

    with colB:
        st.markdown("### 🧩 Shell untersuchen")
        hs = shells_for_layout_box(instance, chosen_layout, chosen_box)
        if not hs:
            st.warning("Keine Shells für diese Box.")
        else:
            shell_id = st.selectbox("Shell wählen", hs, index=0, key="chosen_shell_select")
            sp = shell_params(instance, shell_id)
            smin = int(sp.get("min", 0))
            smax = int(sp.get("max", 0))

            st.write(f"**Shell {shell_id}**")
            st.write(f"- min chars: **{smin}**")
            st.write(f"- max chars: **{smax}**")

            st.markdown("#### 🔎 Filter: Under-/Overfill-Threshold")
            filter_on = st.checkbox(
                "Filter aktivieren (Artikel mit zu starkem Under/Overfill ausblenden)",
                value=False,
                key="filter_on",
            )

            c1, c2 = st.columns(2)
            with c1:
                under_thr = st.slider(
                    "Max. Underfill (%)",
                    min_value=0.0, max_value=100.0, value=15.0, step=1.0,
                    disabled=not filter_on, key="under_thr",
                )
            with c2:
                over_thr = st.slider(
                    "Max. Overfill (%)",
                    min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                    disabled=not filter_on, key="over_thr",
                )

            arts = compatible_articles_for_shell(instance, shell_id)

            # Nur Artikel zeigen die laut Section auf dieser Seite erlaubt sind
            allowed_arts = set()
            for sec in get_sections_for_page(instance, page_id):
                allowed_arts.update(get_articles_for_section(instance, sec))
            arts = [a for a in arts if a in allowed_arts]

            if not arts:
                st.warning("Keine kompatiblen Artikel für diesen Shell gefunden.")
            else:
                rows = []
                for a in arts:
                    L = article_len(instance, a)
                    pr = article_prio(instance, a)
                    secs = get_pages_for_article(instance, a)

                    under_pct = 0.0
                    over_pct = 0.0
                    if smin > 0 and L < smin:
                        under_pct = 100.0 * (smin - L) / smin
                    if smax > 0 and L > smax:
                        over_pct = 100.0 * (L - smax) / smax

                    fits = (L >= smin) and (L <= smax)
                    rows.append(
                        {
                            "article": int(a),
                            "prio": pr,
                            "length": int(L),
                            "sections": ", ".join(map(str, secs)),
                            "fits": fits,
                            "underfill_%": round(under_pct, 2),
                            "overfill_%": round(over_pct, 2),
                        }
                    )

                adf = pd.DataFrame(rows)

                if filter_on:
                    before = len(adf)
                    adf = adf[(adf["underfill_%"] <= under_thr) & (adf["overfill_%"] <= over_thr)]
                    st.caption(f"Filter aktiv: {before - len(adf)} ausgeblendet, {len(adf)} übrig.")

                if adf.empty:
                    st.warning("Keine Artikel nach Filter. Thresholds erhöhen oder Filter deaktivieren.")
                else:
                    prio_order = {"A": 0, "B": 1, "C": 2}
                    adf["_p"] = adf["prio"].map(lambda p: prio_order.get(p, 9))
                    adf["misfit_%"] = adf[["underfill_%", "overfill_%"]].max(axis=1)
                    adf = (
                        adf.sort_values(["fits", "_p", "misfit_%", "length"], ascending=[False, True, True, True])
                        .drop(columns=["_p"])
                    )
                    adf["fits"] = adf["fits"].map(lambda x: "✅" if x else "❌")
                    st.dataframe(adf, use_container_width=True, hide_index=True)

    st.divider()

# ============================================================
# TAB 2: Sections
# ============================================================
with tab_sections:
    st.subheader("📋 Sections")

    pages_all = [int(p) for p in instance.get("pages", [])]

    for pid in pages_all:
        st.markdown(f"### Seite {pid}")
        secs = get_sections_for_page(instance, pid)
        if not secs:
            st.caption("Keine Sections für diese Seite.")
            continue
        for sec in secs:
            arts = get_articles_for_section(instance, sec)
            with st.expander(f"Section {sec}  —  {len(arts)} Artikel", expanded=False):
                if not arts:
                    st.caption("Keine Artikel.")
                else:
                    rows = []
                    for a in arts:
                        rows.append({
                            "artikel": a,
                            "prio":    article_prio(instance, a),
                            "länge":   article_len(instance, a),
                        })
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )
        st.divider()

# ============================================================
# TAB 3: Solution
# ============================================================
with tab_solution:
    st.subheader("✅ Solution Viewer")

    sol_file = st.file_uploader("Solution-Datei (.sol)", type=["sol"], key="sol_upload")

    if sol_file is None:
        st.info("Lade eine .sol-Datei hoch.")
    else:
        # --- Parse solution ---
        sol: Dict[str, float] = {}
        objective = None
        for line in sol_file.read().decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Objective"):
                try:
                    objective = float(line.split("=")[1].strip())
                except Exception:
                    pass
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                sol[parts[0]] = float(parts[1])

        # y_{page}_{layout} = 1  →  layout assigned to page
        page_layout: Dict[int, int] = {}
        for k, v in sol.items():
            if k.startswith("y_") and v == 1:
                p = k.split("_")
                page_layout[int(p[1])] = int(p[2])

        # x_{art}_{page}_{layout}_{box}_{shell} = 1  →  article placed
        assignments: List[Dict] = []
        for k, v in sol.items():
            if k.startswith("x_") and v == 1:
                p = k.split("_")
                assignments.append({
                    "art":    int(p[1]),
                    "page":   int(p[2]),
                    "layout": int(p[3]),
                    "box":    int(p[4]),
                    "shell":  int(p[5]),
                })

        # --- Header metrics ---
        n_placed = len(assignments)
        n_pages_used = len(page_layout)
        pages_sol = sorted(page_layout.keys())

        h1, h2, h3 = st.columns(3)
        h1.metric("Objective", f"{objective:.4f}" if objective is not None else "—")
        h2.metric("Platzierte Artikel", n_placed)
        h3.metric("Seiten in Solution", n_pages_used)

        st.divider()

        # --- Per-page view ---
        for pid in pages_sol:
            layout_id = page_layout[pid]
            page_assignments = [a for a in assignments if a["page"] == pid]
            assigned_boxes = {a["box"] for a in page_assignments}

            all_boxes = get_layout_boxes(instance, layout_id)
            n_filled = len(assigned_boxes)
            n_total  = len(all_boxes)

            # f_page from solution
            f_page = sol.get(f"f_page_{pid}", None)
            fill_str = f"{f_page:.1%}" if f_page is not None else "—"

            with st.expander(
                f"Seite {pid}  ·  Layout {layout_id}  ·  "
                f"{n_filled}/{n_total} Boxen belegt  ·  Füllgrad {fill_str}",
                expanded=True,
            ):
                col_chart, col_table = st.columns([1, 1.4], gap="large")

                with col_chart:
                    # Build df for box_chart, color boxes by status
                    df_layout = layout_rects_df(instance, layout_id)
                    if not df_layout.empty:
                        # Mark each box: filled / empty
                        df_layout["status"] = df_layout["box"].apply(
                            lambda b: "belegt" if b in assigned_boxes else "leer"
                        )
                        # Add article info per box for tooltip
                        box_to_art = {a["box"]: a for a in page_assignments}
                        df_layout["artikel"] = df_layout["box"].apply(
                            lambda b: str(box_to_art[b]["art"]) if b in box_to_art else "—"
                        )
                        df_layout["prio"] = df_layout["box"].apply(
                            lambda b: article_prio(instance, box_to_art[b]["art"]) if b in box_to_art else "—"
                        )
                        df_layout["länge"] = df_layout["box"].apply(
                            lambda b: article_len(instance, box_to_art[b]["art"]) if b in box_to_art else 0
                        )
                        df_layout["shell"] = df_layout["box"].apply(
                            lambda b: str(box_to_art[b]["shell"]) if b in box_to_art else "—"
                        )
                        # Fill % per box
                        def box_fill(b):
                            if b not in box_to_art:
                                return "—"
                            a = box_to_art[b]
                            sp = shell_params(instance, a["shell"])
                            smax = sp.get("max", 0)
                            if smax == 0:
                                return "—"
                            L = article_len(instance, a["art"])
                            over  = sol.get(f"delta_over_{a['layout']}_{pid}_{b}", 0)
                            under = sol.get(f"delta_under_{a['layout']}_{pid}_{b}", 0)
                            pct = L / smax
                            flag = " ⚠️" if over or under else ""
                            return f"{pct:.0%}{flag}"
                        df_layout["fill"] = df_layout["box"].apply(box_fill)

                        # Farbschema — direkt aus MILP-Variablen:
                        # leer (kein Artikel)          → rot
                        # e_over=1 oder e_under=1      → rot  (Artikel platziert aber wie leer behandelt)
                        # delta_under=1, e_under=0     → lila (toleriertes Underfill)
                        # delta_over=1,  e_over=0      → blau (toleriertes Overfill)
                        # sonst                        → grün (passt)
                        # Key-Format: {var}_{page}_{layout}_{box}
                        def box_color(row):
                            b = row["box"]
                            if b not in box_to_art:
                                return "#ef4444"
                            e_ov = sol.get(f"e_over_{pid}_{layout_id}_{b}",  0)
                            e_un = sol.get(f"e_under_{pid}_{layout_id}_{b}", 0)
                            if e_ov or e_un:
                                return "#ef4444"  # wie leer → rot
                            d_ov = sol.get(f"delta_over_{pid}_{layout_id}_{b}",  0)
                            d_un = sol.get(f"delta_under_{pid}_{layout_id}_{b}", 0)
                            if d_un:
                                return "#a855f7"  # toleriertes Underfill → lila
                            if d_ov:
                                return "#60a5fa"  # toleriertes Overfill → blau
                            return "#86efac"      # passt → grün

                        df_layout["color"] = df_layout.apply(box_color, axis=1)

                        # Label: "Art {id}\nPrio {p}\n{fill}"
                        def box_label(row):
                            b = row["box"]
                            if b not in box_to_art:
                                return "leer"
                            a = box_to_art[b]
                            return str(a["art"])

                        def box_sublabel(row):
                            b = row["box"]
                            if b not in box_to_art:
                                return ""
                            a = box_to_art[b]
                            pr = article_prio(instance, a["art"])
                            return f"Prio {pr} · {row['fill']}"

                        df_layout["label"]    = df_layout.apply(box_label, axis=1)
                        df_layout["sublabel"] = df_layout.apply(box_sublabel, axis=1)

                        # Chart über preview-style: kein Achsen-Overhead, einfaches Koordinatensystem
                        W, H = canvas_wh(instance)
                        MX, MY = MARGIN_X, MARGIN_Y
                        FW = W + 2 * MX
                        FH = H + 2 * MY
                        width_px  = 300
                        height_px = int(width_px * (FH / FW))

                        NO_AXIS = _no_axis()
                        xs = alt.Scale(domain=[0, FW], nice=False)
                        ys = alt.Scale(domain=[0, FH], nice=False, reverse=True)

                        sdf = _shift_df(df_layout)
                        sdf["plx"] = (sdf["x0"] + sdf["x1"]) / 2
                        sdf["ply"] = (sdf["y0"] + sdf["y1"]) / 2
                        sdf["psy"] = sdf["y0"] + 0.12

                        def props():
                            return dict(width=width_px, height=height_px)

                        # Hintergrund-Layer (Papier + Margin + Satzspiegel + Gitter)
                        base_layers, _, _, _ = _newspaper_layers(instance, width_px)

                        rects = alt.Chart(sdf).mark_rect(
                            stroke="#1e293b", strokeWidth=1.2
                        ).encode(
                            x=alt.X("x0:Q",  axis=NO_AXIS, scale=xs),
                            x2="x1:Q",
                            y=alt.Y("y0:Q",  axis=NO_AXIS, scale=ys),
                            y2="y1:Q",
                            color=alt.Color("color:N", scale=None, legend=None),
                            tooltip=[
                                alt.Tooltip("box:N",      title="Box"),
                                alt.Tooltip("shell:N",    title="Shell"),
                                alt.Tooltip("artikel:N",  title="Artikel"),
                                alt.Tooltip("prio:N",     title="Prio"),
                                alt.Tooltip("länge:Q",    title="Länge"),
                                alt.Tooltip("fill:N",     title="Füllgrad"),
                            ],
                        ).properties(**props())

                        main_labels = alt.Chart(sdf).mark_text(
                            fontSize=13, fontWeight="bold", color="#1e293b",
                            baseline="middle", align="center",
                        ).encode(
                            x=alt.X("plx:Q", axis=NO_AXIS, scale=xs),
                            y=alt.Y("ply:Q", axis=NO_AXIS, scale=ys),
                            text=alt.Text("label:N"),
                        ).properties(**props())

                        sub_labels = alt.Chart(sdf).mark_text(
                            fontSize=8, color="#1e293b", baseline="top", align="center",
                        ).encode(
                            x=alt.X("plx:Q", axis=NO_AXIS, scale=xs),
                            y=alt.Y("psy:Q", axis=NO_AXIS, scale=ys),
                            text=alt.Text("sublabel:N"),
                        ).properties(**props())

                        chart = alt.layer(*base_layers, rects, sub_labels, main_labels)
                        st.altair_chart(chart.configure_view(stroke=None), use_container_width=False)

                        # Legende
                        st.caption(
                            "🟢 passt · 🔵 Overfill (toleriert) · 🟣 Underfill (toleriert) · "
                            "🔴 leer oder e-Variabel aktiv (Artikel wie leer behandelt)"
                        )

                with col_table:
                    # Detail table for this page
                    rows = []
                    for a in sorted(page_assignments, key=lambda x: x["box"]):
                        sp = shell_params(instance, a["shell"])
                        smin = int(sp.get("min", 0))
                        smax = int(sp.get("max", 0))
                        L    = article_len(instance, a["art"])
                        fits = smin <= L <= smax
                        under = round(100 * (smin - L) / smin, 1) if L < smin and smin > 0 else 0.0
                        over  = round(100 * (L - smax) / smax, 1) if L > smax and smax > 0 else 0.0
                        d_over  = int(sol.get(f"delta_over_{pid}_{layout_id}_{a['box']}", 0))
                        d_under = int(sol.get(f"delta_under_{pid}_{layout_id}_{a['box']}", 0))
                        rows.append({
                            "box":       a["box"],
                            "shell":     a["shell"],
                            "artikel":   a["art"],
                            "prio":      article_prio(instance, a["art"]),
                            "länge":     L,
                            "shell min": smin,
                            "shell max": smax,
                            "fits":      "✅" if fits else "❌",
                            "over_%":    over,
                            "under_%":   under,
                        })
                    # Empty boxes
                    for b in all_boxes:
                        if b not in assigned_boxes:
                            rows.append({
                                "box": b, "shell": "—", "artikel": "—",
                                "prio": "—", "länge": 0,
                                "shell min": 0, "shell max": 0,
                                "fits": "—", "over_%": 0.0, "under_%": 0.0,
                            })
                    if rows:
                        tdf = pd.DataFrame(rows).sort_values("box")
                        st.dataframe(tdf, use_container_width=True, hide_index=True)