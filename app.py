import json
import math
import random
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# PAGE GEOMETRY (from instance["canvas"])
# ============================================================
# With real box geometries, a synthetic margin is usually not needed.
PAGE_MARGIN = 0.0
# ============================================================


# -----------------------------
# Instance helpers (robust)
# -----------------------------
def get_page_layouts(instance: dict, page_id: int) -> List[int]:
    layouts = instance.get("layouts_pages", {}).get(str(page_id), [])
    return [int(x) for x in layouts]


def get_layout_boxes(instance: dict, layout_id: int) -> List[int]:
    boxes = instance.get("box_layouts", {}).get(str(layout_id), [])
    return [int(b) for b in boxes]


def hulls_for_layout_box(instance: dict, layout_id: int, box_id: int) -> List[int]:
    return [
        int(h)
        for h in instance.get("hull_layout_box", {})
        .get(str(layout_id), {})
        .get(str(box_id), [])
    ]


def hull_params(instance: dict, hull_id: int) -> Dict[str, float]:
    return instance.get("hull_params", {}).get(str(hull_id), {"min": 0, "max": 0})


def article_len(instance: dict, art_id: int) -> int:
    return int(instance.get("article_length", {}).get(str(art_id), 0))


def article_prio(instance: dict, art_id: int) -> str:
    return str(instance.get("article_priority", {}).get(str(art_id), "?"))


def canvas_wh(instance: dict) -> Tuple[float, float]:
    c = instance.get("canvas", {})
    w = float(c.get("w", 210.0))
    h = float(c.get("h", 297.0))
    return w, h


def get_box_geometry(instance: dict, layout_id: int, box_id: int) -> Optional[Dict[str, float]]:
    g = (
        instance.get("geometry_layout_box", {})
        .get(str(layout_id), {})
        .get(str(box_id))
    )
    if not g:
        return None
    return {
        "x": float(g.get("x", 0.0)),
        "y": float(g.get("y", 0.0)),
        "w": float(g.get("w", 0.0)),
        "h": float(g.get("h", 0.0)),
        "area": float(g.get("area", 0.0)),
    }


# -----------------------------
# FIXED: hull->articles mapping builder
# -----------------------------
def build_hull_to_articles(instance: dict) -> Dict[int, List[int]]:
    """
    hull_article:
      keys = article_ids
      values = list of hull_ids
    We invert: hull -> [articles]
    Additionally validate that article_id is in instance["article"].
    """
    valid_articles = set(int(a) for a in instance.get("article", []))
    raw = instance.get("hull_article", {})

    hull_to_articles: Dict[int, List[int]] = {}

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    for ak, hv in raw.items():
        a = to_int(ak)
        if a is None or a not in valid_articles:
            continue
        if not isinstance(hv, list):
            continue
        for h in hv:
            hi = to_int(h)
            if hi is None:
                continue
            hull_to_articles.setdefault(hi, []).append(a)

    for h in list(hull_to_articles.keys()):
        hull_to_articles[h] = sorted(set(hull_to_articles[h]))

    return hull_to_articles


def compatible_articles_for_hull(instance: dict, hull_id: int) -> List[int]:
    """
    Uses prebuilt mapping in session_state.
    Mapping is rebuilt when a new JSON is loaded (see sidebar signature).
    """
    if "_hull_to_articles" not in st.session_state:
        st.session_state["_hull_to_articles"] = build_hull_to_articles(instance)

    valid_articles = set(int(a) for a in instance.get("article", []))
    m = st.session_state["_hull_to_articles"]
    return [int(a) for a in m.get(int(hull_id), []) if int(a) in valid_articles]


# -----------------------------
# Build rectangles dataframe for Altair (REAL GEOMETRY)
# -----------------------------
def layout_rects_df(instance: dict, layout_id: int, y_origin: str = "top") -> pd.DataFrame:
    """
    Uses instance["geometry_layout_box"][layout][box] with real coordinates (mm).

    y_origin:
      - "top": assumes (0,0) at top-left (y down). We flip for plotting.
      - "bottom": assumes (0,0) at bottom-left (y up). No flip.
    """
    W, H = canvas_wh(instance)

    boxes = get_layout_boxes(instance, layout_id)
    if not boxes:
        return pd.DataFrame(
            columns=["layout", "box", "x0", "x1", "y0", "y1", "w", "h", "area", "shells", "num_hulls"]
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

        if y_origin == "top":
            # convert top-origin y-down coords to bottom-origin y-up for plotting
            y_plot = H - (y + h)
        else:
            y_plot = y

        hs = hulls_for_layout_box(instance, layout_id, b)
        rows.append(
            {
                "layout": int(layout_id),
                "box": int(b),
                "x0": float(x),
                "x1": float(x + w),
                "y0": float(y_plot),
                "y1": float(y_plot + h),
                "w": float(w),
                "h": float(h),
                "area": float(geom["area"] if geom["area"] else w * h),
                "shells": ", ".join(map(str, hs)),
                "num_hulls": int(len(hs)),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["x0", "x1", "y0", "y1"]).reset_index(drop=True)
    return df


# -----------------------------
# Plotting (fix for Vega crash)
# -----------------------------
def _no_axis():
    # Never axis=None (can trigger the forEach crash in some Streamlit/Altair combos)
    return alt.Axis(labels=False, ticks=False, domain=False, grid=False, title=None)


def preview_chart(df: pd.DataFrame, instance: dict, width_px=180, active=False):
    if df is None or df.empty:
        return None

    W, H = canvas_wh(instance)
    height_px = int(width_px * (H / W))
    NO_AXIS = _no_axis()

    border_df = pd.DataFrame([{"bx0": 0.0, "bx1": W, "by0": 0.0, "by1": H}])
    border = (
        alt.Chart(border_df)
        .mark_rect(fillOpacity=0, stroke="black", strokeWidth=1.6)
        .encode(
            x=alt.X("bx0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, W], nice=False)),
            x2="bx1:Q",
            y=alt.Y("by0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, H], nice=False)),
            y2="by1:Q",
        )
        .properties(width=width_px, height=height_px)
    )

    base = (
        alt.Chart(df)
        .mark_rect(stroke="black", strokeWidth=0.8)
        .encode(
            x=alt.X("x0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, W], nice=False)),
            x2="x1:Q",
            y=alt.Y("y0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, H], nice=False)),
            y2="y1:Q",
            color=alt.value("#dbeafe" if active else "#f3f4f6"),
            tooltip=[alt.Tooltip("box:N", title="Box"), alt.Tooltip("num_hulls:Q", title="#Hulls")],
        )
        .properties(width=width_px, height=height_px)
    )

    return (border + base).configure_view(stroke=None)


def box_chart(df: pd.DataFrame, instance: dict, width_px=460):
    if df is None or df.empty:
        return None

    W, H = canvas_wh(instance)
    height_px = int(width_px * (H / W))
    NO_AXIS = _no_axis()

    border_df = pd.DataFrame([{"bx0": 0.0, "bx1": W, "by0": 0.0, "by1": H}])
    border = alt.Chart(border_df).mark_rect(fillOpacity=0, stroke="black", strokeWidth=2).encode(
        x=alt.X("bx0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, W], nice=False)),
        x2="bx1:Q",
        y=alt.Y("by0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, H], nice=False)),
        y2="by1:Q",
    )

    rects = alt.Chart(df).mark_rect(stroke="black", strokeWidth=1).encode(
        x=alt.X("x0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, W], nice=False)),
        x2="x1:Q",
        y=alt.Y("y0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, H], nice=False)),
        y2="y1:Q",
        color=alt.value("#e5e7eb"),
        tooltip=[
            alt.Tooltip("box:N", title="Box"),
            alt.Tooltip("num_hulls:Q", title="#Hulls"),
            alt.Tooltip("shells:N", title="Hulls"),
            alt.Tooltip("w:Q", title="w (mm)"),
            alt.Tooltip("h:Q", title="h (mm)"),
        ],
    )

    labels = alt.Chart(df).mark_text(fontSize=12, dy=-6).encode(
        x=alt.X("x0:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, W], nice=False)),
        y=alt.Y("y1:Q", axis=NO_AXIS, scale=alt.Scale(domain=[0, H], nice=False)),
        text=alt.Text("box:N"),
    )

    return (border + rects + labels).properties(width=width_px, height=height_px).configure_view(stroke=None)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Newspaper Instance Explorer (Canvas)", layout="wide")

st.title("🗞️ Newspaper Instance Explorer")
st.caption("Interaktive Ansicht: Seiten → Layouts → Boxen → Shells → kompatible Artikel (+ Under/Overfill Filter)")

with st.sidebar:
    st.header("📦 Instanz laden")
    up = st.file_uploader("JSON-Datei (deine Instanz)", type=["json"])
    if up is None:
        st.info("Lade eine JSON-Datei hoch.")
        st.stop()

    instance = json.load(up)
    st.success("Instanz geladen ✅")

    # Reset cached mapping when new JSON is loaded
    inst_sig = (getattr(up, "name", None), getattr(up, "size", None))
    if st.session_state.get("_instance_sig") != inst_sig:
        st.session_state["_instance_sig"] = inst_sig
        st.session_state.pop("_hull_to_articles", None)

    W, H = canvas_wh(instance)
    st.caption(f"Canvas: **{W:.1f} × {H:.1f} mm**")

    y_origin = st.selectbox("Koordinaten-Ursprung (y)", ["top", "bottom"], index=0)

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

# Session state
if "chosen_layout" not in st.session_state:
    st.session_state["chosen_layout"] = int(layouts[0])
if "chosen_box" not in st.session_state:
    st.session_state["chosen_box"] = None

# Layout grid
st.subheader(f"Seite {page_id}: Layout Vorschau")

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

            df_prev = layout_rects_df(instance, lid, y_origin=y_origin)
            prev_chart = preview_chart(df_prev, instance, width_px=180, active=is_active)
            if prev_chart is None:
                st.warning("Leeres Layout (keine Box-Daten).")
            else:
                st.altair_chart(prev_chart, use_container_width=False)

st.divider()

chosen_layout = int(st.session_state["chosen_layout"])
st.subheader(f"Layout {chosen_layout}: Boxen untersuchen")

df = layout_rects_df(instance, chosen_layout, y_origin=y_origin)
main_chart = box_chart(df, instance, width_px=460)
if main_chart is None:
    st.error("Keine Box-Daten für dieses Layout.")
    st.stop()
st.altair_chart(main_chart, use_container_width=False)

box_table = df[["box", "num_hulls", "shells", "w", "h", "area"]].sort_values("box").reset_index(drop=True)
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

# Box / Hull explorer
colA, colB = st.columns([1, 1.3], gap="large")

with colA:
    st.markdown("### 🔳 Box-Details")
    st.write(f"**Box:** {chosen_box}")

    geom = get_box_geometry(instance, chosen_layout, chosen_box)
    if geom:
        st.write(f"**Geometrie (mm):** x={geom['x']:.2f}, y={geom['y']:.2f}, w={geom['w']:.2f}, h={geom['h']:.2f}")

    hs = hulls_for_layout_box(instance, chosen_layout, chosen_box)
    st.write(f"**Shells in dieser Box:** {hs if hs else '—'}")

    if hs:
        hull_rows = []
        for h in hs:
            hp = hull_params(instance, h)
            hull_rows.append(
                {
                    "shell": h,
                    "min": int(hp.get("min", 0)),
                    "max": int(hp.get("max", 0)),
                    "#articles": len(compatible_articles_for_hull(instance, h)),
                }
            )
        hull_df = pd.DataFrame(hull_rows).sort_values("shell")
        st.dataframe(hull_df, use_container_width=True, hide_index=True)

with colB:
    st.markdown("### 🧩 Shell untersuchen")
    hs = hulls_for_layout_box(instance, chosen_layout, chosen_box)
    if not hs:
        st.warning("Keine Shells für diese Box.")
    else:
        hull_id = st.selectbox("Hull wählen", hs, index=0, key="chosen_hull_select")
        hp = hull_params(instance, hull_id)
        hmin = int(hp.get("min", 0))
        hmax = int(hp.get("max", 0))

        st.write(f"**Hull {hull_id}**")
        st.write(f"- min chars: **{hmin}**")
        st.write(f"- max chars: **{hmax}**")

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
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                disabled=not filter_on,
                key="under_thr",
            )
        with c2:
            over_thr = st.slider(
                "Max. Overfill (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                disabled=not filter_on,
                key="over_thr",
            )

        arts = compatible_articles_for_hull(instance, hull_id)
        if not arts:
            st.warning("Keine kompatiblen Artikel für diesen Hull gefunden (hull_article leer / keine Zuordnung).")
        else:
            rows = []
            for a in arts:
                L = article_len(instance, a)
                pr = article_prio(instance, a)

                under_pct = 0.0
                over_pct = 0.0
                if hmin > 0 and L < hmin:
                    under_pct = 100.0 * (hmin - L) / hmin
                if hmax > 0 and L > hmax:
                    over_pct = 100.0 * (L - hmax) / hmax

                fits = (L >= hmin) and (L <= hmax)
                rows.append(
                    {
                        "article": int(a),
                        "prio": pr,
                        "length": int(L),
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
