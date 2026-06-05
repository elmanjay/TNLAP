import json
import math
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import streamlit as st
import altair as alt

DEFAULT_GRID_W = 6
DEFAULT_GRID_H = 10

def get_page_layouts(instance, page_id):
    return [int(x) for x in instance.get("layouts_pages", {}).get(str(page_id), [])]

def get_layout_boxes(instance, layout_id):
    return [int(b) for b in instance.get("box_layouts", {}).get(str(layout_id), [])]

def shells_for_layout_box(instance, layout_id, box_id):
    return [int(h) for h in instance.get("shells_layout_box", {}).get(str(layout_id), {}).get(str(box_id), [])]

def shell_params(instance, shell_id):
    return instance.get("shell_params", {}).get(str(shell_id), {"min": 0, "max": 0})

def article_len(instance, art_id):
    return int(instance.get("article_length", {}).get(str(art_id), 0))

def article_prio(instance, art_id):
    return str(instance.get("article_priority", {}).get(str(art_id), "?"))

def canvas_wh(instance):
    geom_all = instance.get("geometry_layout_box", {})
    max_x, max_y = 0.0, 0.0
    for boxes in geom_all.values():
        for g in boxes.values():
            max_x = max(max_x, float(g.get("x", 0)) + float(g.get("w", 0)))
            max_y = max(max_y, float(g.get("y", 0)) + float(g.get("h", 0)))
    return (max_x or float(DEFAULT_GRID_W)), (max_y or float(DEFAULT_GRID_H))

def get_box_geometry(instance, layout_id, box_id):
    g = instance.get("geometry_layout_box", {}).get(str(layout_id), {}).get(str(box_id))
    if not g: return None
    w, h = float(g.get("w", 0)), float(g.get("h", 0))
    return {"x": float(g.get("x", 0)), "y": float(g.get("y", 0)), "w": w, "h": h,
            "area": w*h, "character": float(g.get("character", g.get("area", w*h)))}

def build_shell_to_articles(instance):
    valid = set(int(a) for a in instance.get("article", []))
    raw = instance.get("shells_article", {})
    s2a = {}
    for ak, sv in raw.items():
        try: a = int(ak)
        except: continue
        if a not in valid or not isinstance(sv, list): continue
        for s in sv:
            try: si = int(s)
            except: continue
            s2a.setdefault(si, []).append(a)
    for s in s2a: s2a[s] = sorted(set(s2a[s]))
    return s2a

def compatible_articles_for_shell(instance, shell_id):
    if "_shell_to_articles" not in st.session_state:
        st.session_state["_shell_to_articles"] = build_shell_to_articles(instance)
    valid = set(int(a) for a in instance.get("article", []))
    m = st.session_state["_shell_to_articles"]
    return [int(a) for a in m.get(int(shell_id), []) if int(a) in valid]

def get_sections_for_page(instance, page_id):
    return [int(s) for s in instance.get("sections_page", {}).get(str(page_id), [])]

def get_articles_for_section(instance, section_id):
    return [int(a) for a in instance.get("article_sections", {}).get(str(section_id), [])]

def get_pages_for_article(instance, art_id):
    pages = []
    for page_id in instance.get("pages", []):
        for sec in get_sections_for_page(instance, int(page_id)):
            if int(art_id) in get_articles_for_section(instance, sec):
                pages.append(int(page_id))
    return sorted(set(pages))

def layout_rects_df(instance, layout_id):
    boxes = get_layout_boxes(instance, layout_id)
    if not boxes:
        return pd.DataFrame(columns=["layout","box","x0","x1","y0","y1","w","h","area","character","shells","num_shells"])
    rows = []
    for b in boxes:
        geom = get_box_geometry(instance, layout_id, b)
        if geom is None: continue
        hs = shells_for_layout_box(instance, layout_id, b)
        rows.append({"layout": int(layout_id), "box": int(b),
                     "x0": float(geom["x"]), "x1": float(geom["x"]+geom["w"]),
                     "y0": float(geom["y"]), "y1": float(geom["y"]+geom["h"]),
                     "w": float(geom["w"]), "h": float(geom["h"]),
                     "area": float(geom["area"]), "character": float(geom["character"]),
                     "shells": ", ".join(map(str, hs)), "num_shells": int(len(hs))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["x0","x1","y0","y1"]).reset_index(drop=True)
    return df

MARGIN_X = 0.35
MARGIN_Y = 0.45
N_COLS   = 6

def _no_axis():
    return alt.Axis(labels=False, ticks=False, domain=False, grid=False, title=None)

def _newspaper_layers(instance, width_px):
    W, H = canvas_wh(instance)
    MX, MY = MARGIN_X, MARGIN_Y
    FW = W + 2*MX
    FH = H + 2*MY
    height_px = int(width_px * (FH / FW))
    NO_AXIS = _no_axis()
    xs = alt.Scale(domain=[0, FW], nice=False)
    ys = alt.Scale(domain=[0, FH], nice=False, reverse=True)
    def xenc(f): return alt.X(f"{f}:Q", axis=NO_AXIS, scale=xs)
    def yenc(f): return alt.Y(f"{f}:Q", axis=NO_AXIS, scale=ys)
    bg_df = pd.DataFrame([{"x0":0,"x1":FW,"y0":0,"y1":FH}])
    bg = alt.Chart(bg_df).mark_rect(fill="#fdf6e3",stroke=None).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(width=width_px,height=height_px)
    margin_rects = [{"x0":0,"x1":MX,"y0":0,"y1":FH},{"x0":MX+W,"x1":FW,"y0":0,"y1":FH},
                    {"x0":MX,"x1":MX+W,"y0":0,"y1":MY},{"x0":MX,"x1":MX+W,"y0":MY+H,"y1":FH}]
    margin_layer = alt.Chart(pd.DataFrame(margin_rects)).mark_rect(fill="#e8dfc8",stroke=None).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(width=width_px,height=height_px)
    ss_df = pd.DataFrame([{"x0":MX,"x1":MX+W,"y0":MY,"y1":MY+H}])
    ss_fill = alt.Chart(ss_df).mark_rect(fill="#ffffff",stroke=None).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(width=width_px,height=height_px)
    col_w = W/N_COLS
    col_lines = [{"x0":MX+i*col_w,"x1":MX+i*col_w,"y0":MY,"y1":MY+H} for i in range(1,N_COLS)]
    col_grid = alt.Chart(pd.DataFrame(col_lines)).mark_rule(stroke="#b0a090",strokeWidth=0.6,strokeDash=[3,4]).encode(x=alt.X("x0:Q",scale=xs),y=alt.Y("y0:Q",scale=ys),y2="y1:Q").properties(width=width_px,height=height_px)
    page_border = alt.Chart(bg_df).mark_rect(fillOpacity=0,stroke="#888888",strokeWidth=0.8).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(width=width_px,height=height_px)
    ss_border   = alt.Chart(ss_df).mark_rect(fillOpacity=0,stroke="#333333",strokeWidth=1.2).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(width=width_px,height=height_px)
    return [bg, margin_layer, ss_fill, col_grid, page_border, ss_border], FW, FH, height_px

def _shift_df(df):
    if df.empty: return df
    out = df.copy()
    out["x0"] += MARGIN_X; out["x1"] += MARGIN_X
    out["y0"] += MARGIN_Y; out["y1"] += MARGIN_Y
    return out

def preview_chart(df, instance, width_px=180, active=False):
    if df is None or df.empty: return None
    base_layers, FW, FH, height_px = _newspaper_layers(instance, width_px)
    NO_AXIS = _no_axis()
    xs = alt.Scale(domain=[0, FW], nice=False)
    ys = alt.Scale(domain=[0, FH], nice=False, reverse=True)
    sdf = _shift_df(df)
    boxes = alt.Chart(sdf).mark_rect(stroke="#555555",strokeWidth=0.7).encode(
        x=alt.X("x0:Q",axis=NO_AXIS,scale=xs), x2="x1:Q",
        y=alt.Y("y0:Q",axis=NO_AXIS,scale=ys), y2="y1:Q",
        color=alt.value("#bfdbfe" if active else "#e2e8f0"),
        tooltip=[alt.Tooltip("box:N",title="Box"),alt.Tooltip("num_shells:Q",title="#Shells")],
    ).properties(width=width_px,height=height_px)
    return alt.layer(*base_layers, boxes).configure_view(stroke=None)

def box_chart(df, instance, width_px=460):
    if df is None or df.empty: return None
    W, H = canvas_wh(instance)
    ROW_SCALE = 100/H; COL_SCALE = N_COLS/W
    MX, MY = MARGIN_X, MARGIN_Y
    LEFT_AXIS_W = 0.55; TOP_HEADER_H = 0.55
    FW = LEFT_AXIS_W + W + 2*MX; FH = TOP_HEADER_H + H + 2*MY
    height_px = int(width_px*(FH/FW))
    NO_AXIS = _no_axis()
    xs = alt.Scale(domain=[0,FW],nice=False)
    ys = alt.Scale(domain=[0,FH],nice=False,reverse=True)
    OX = LEFT_AXIS_W+MX; OY = TOP_HEADER_H+MY
    def xenc(f): return alt.X(f"{f}:Q",axis=NO_AXIS,scale=xs)
    def yenc(f): return alt.Y(f"{f}:Q",axis=NO_AXIS,scale=ys)
    def props(): return dict(width=width_px,height=height_px)
    sx0=LEFT_AXIS_W+MX; sx1=LEFT_AXIS_W+MX+W; sy0=TOP_HEADER_H+MY; sy1=TOP_HEADER_H+MY+H
    bg_df=pd.DataFrame([{"x0":0,"x1":FW,"y0":0,"y1":FH}])
    bg=alt.Chart(bg_df).mark_rect(fill="#fdf6e3",stroke=None).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(**props())
    margin_rects=[{"x0":LEFT_AXIS_W,"x1":sx0,"y0":TOP_HEADER_H,"y1":FH},{"x0":sx1,"x1":FW,"y0":TOP_HEADER_H,"y1":FH},
                  {"x0":sx0,"x1":sx1,"y0":TOP_HEADER_H,"y1":sy0},{"x0":sx0,"x1":sx1,"y0":sy1,"y1":FH}]
    margin_layer=alt.Chart(pd.DataFrame(margin_rects)).mark_rect(fill="#e8dfc8",stroke=None).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(**props())
    ss_df=pd.DataFrame([{"x0":sx0,"x1":sx1,"y0":sy0,"y1":sy1}])
    ss_fill=alt.Chart(ss_df).mark_rect(fill="#ffffff",stroke=None).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(**props())
    col_unit=W/N_COLS
    col_lines=[{"x0":sx0+i*col_unit,"x1":sx0+i*col_unit,"y0":sy0,"y1":sy1} for i in range(1,N_COLS)]
    col_grid=alt.Chart(pd.DataFrame(col_lines)).mark_rule(stroke="#b0a090",strokeWidth=0.6,strokeDash=[3,4]).encode(x=alt.X("x0:Q",scale=xs),y=alt.Y("y0:Q",scale=ys),y2="y1:Q").properties(**props())
    page_border=alt.Chart(bg_df).mark_rect(fillOpacity=0,stroke="#888888",strokeWidth=0.8).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(**props())
    ss_border=alt.Chart(ss_df).mark_rect(fillOpacity=0,stroke="#333333",strokeWidth=1.2).encode(xenc("x0"),alt.X2("x1:Q"),yenc("y0"),alt.Y2("y1:Q")).properties(**props())
    HEADER_FILL="#334155"; HEADER_STROKE="#1e293b"; HEADER_TEXT="#f8fafc"
    HEADER_TOP=TOP_HEADER_H*0.08; HEADER_BOT=TOP_HEADER_H*0.92
    col_header_rects=[]; col_header_labels=[]
    for i in range(N_COLS):
        cx0=sx0+i*col_unit+0.02; cx1=sx0+(i+1)*col_unit-0.02
        col_header_rects.append({"x0":cx0,"x1":cx1,"y0":HEADER_TOP,"y1":HEADER_BOT,"col":i+1})
        col_header_labels.append({"lx":(cx0+cx1)/2,"ly":(HEADER_TOP+HEADER_BOT)/2,"label":f"Col. {i+1}"})
    col_hdr_rects=alt.Chart(pd.DataFrame(col_header_rects)).mark_rect(fill=HEADER_FILL,stroke=HEADER_STROKE,strokeWidth=0.8,cornerRadius=2).encode(x=alt.X("x0:Q",axis=NO_AXIS,scale=xs),x2="x1:Q",y=alt.Y("y0:Q",axis=NO_AXIS,scale=ys),y2="y1:Q",tooltip=[alt.Tooltip("col:Q",title="Spalte")]).properties(**props())
    col_hdr_text=alt.Chart(pd.DataFrame(col_header_labels)).mark_text(fontSize=10,fontWeight="bold",color=HEADER_TEXT,baseline="middle",align="center").encode(x=alt.X("lx:Q",axis=NO_AXIS,scale=xs),y=alt.Y("ly:Q",axis=NO_AXIS,scale=ys),text=alt.Text("label:N")).properties(**props())
    hdr_line=alt.Chart(pd.DataFrame([{"x0":sx0,"x1":sx1,"y":sy0}])).mark_rule(stroke="#334155",strokeWidth=1.5).encode(x=alt.X("x0:Q",scale=xs),x2="x1:Q",y=alt.Y("y:Q",scale=ys)).properties(**props())
    ROW_TICKS=list(range(0,101,10))
    TICK_X1=LEFT_AXIS_W-0.04; TICK_X0=LEFT_AXIS_W-0.12; LABEL_X=LEFT_AXIS_W-0.14
    row_ticks_data=[{"x0":TICK_X0,"x1":TICK_X1,"y":OY+r/ROW_SCALE} for r in ROW_TICKS]
    row_labels_data=[{"lx":LABEL_X,"ly":OY+r/ROW_SCALE,"label":str(r)} for r in ROW_TICKS]
    row_ticks=alt.Chart(pd.DataFrame(row_ticks_data)).mark_rule(stroke="#64748b",strokeWidth=1.0).encode(x=alt.X("x0:Q",scale=xs),x2="x1:Q",y=alt.Y("y:Q",scale=ys)).properties(**props())
    row_labels=alt.Chart(pd.DataFrame(row_labels_data)).mark_text(fontSize=9,color="#475569",baseline="middle",align="right").encode(x=alt.X("lx:Q",axis=NO_AXIS,scale=xs),y=alt.Y("ly:Q",axis=NO_AXIS,scale=ys),text=alt.Text("label:N")).properties(**props())
    axis_line=alt.Chart(pd.DataFrame([{"x":sx0-MX*0.5,"y0":sy0,"y1":sy1}])).mark_rule(stroke="#94a3b8",strokeWidth=0.8).encode(x=alt.X("x:Q",scale=xs),y=alt.Y("y0:Q",scale=ys),y2="y1:Q").properties(**props())
    hgrid=alt.Chart(pd.DataFrame([{"x0":sx0,"x1":sx1,"y":OY+r/ROW_SCALE} for r in ROW_TICKS[1:-1]])).mark_rule(stroke="#d1cbc0",strokeWidth=0.4,strokeDash=[2,3]).encode(x=alt.X("x0:Q",scale=xs),x2="x1:Q",y=alt.Y("y:Q",scale=ys)).properties(**props())
    sdf=df.copy()
    sdf["col_start"]=(sdf["x0"]*COL_SCALE).round().astype(int)+1
    sdf["col_end"]=(sdf["x1"]*COL_SCALE).round().astype(int)
    sdf["row_start"]=(sdf["y0"]*ROW_SCALE).round().astype(int)+1
    sdf["row_end"]=(sdf["y1"]*ROW_SCALE).round().astype(int)
    sdf["n_cols"]=(sdf["w"]*COL_SCALE).round().astype(int)
    sdf["n_rows"]=(sdf["h"]*ROW_SCALE).round().astype(int)
    sdf["col_label"]=sdf["n_cols"].astype(str)+" Col. / "+sdf["n_rows"].astype(str)+" R."
    sdf["px0"]=sdf["x0"]+OX; sdf["px1"]=sdf["x1"]+OX
    sdf["py0"]=sdf["y0"]+OY; sdf["py1"]=sdf["y1"]+OY
    sdf["plx"]=(sdf["px0"]+sdf["px1"])/2; sdf["ply"]=(sdf["py0"]+sdf["py1"])/2
    sdf["psy"]=sdf["py0"]+0.10
    box_rects=alt.Chart(sdf).mark_rect(stroke="#1e293b",strokeWidth=1.4).encode(x=alt.X("px0:Q",axis=NO_AXIS,scale=xs),x2="px1:Q",y=alt.Y("py0:Q",axis=NO_AXIS,scale=ys),y2="py1:Q",color=alt.value("#dbeafe"),tooltip=[alt.Tooltip("box:N",title="Box"),alt.Tooltip("col_start:Q",title="Col. von"),alt.Tooltip("col_end:Q",title="Col. bis"),alt.Tooltip("n_cols:Q",title="Breite (Col.)"),alt.Tooltip("row_start:Q",title="Zeile von"),alt.Tooltip("row_end:Q",title="Zeile bis"),alt.Tooltip("n_rows:Q",title="Höhe (Zeilen)"),alt.Tooltip("num_shells:Q",title="#Shells"),alt.Tooltip("character:Q",title="capacity (chars)")]).properties(**props())
    box_num=alt.Chart(sdf).mark_text(fontSize=14,fontWeight="bold",color="#1e3a5f",baseline="middle",align="center").encode(x=alt.X("plx:Q",axis=NO_AXIS,scale=xs),y=alt.Y("ply:Q",axis=NO_AXIS,scale=ys),text=alt.Text("box:N")).properties(**props())
    box_sub=alt.Chart(sdf).mark_text(fontSize=8,color="#475569",baseline="top",align="center").encode(x=alt.X("plx:Q",axis=NO_AXIS,scale=xs),y=alt.Y("psy:Q",axis=NO_AXIS,scale=ys),text=alt.Text("col_label:N")).properties(**props())
    return alt.layer(bg,margin_layer,ss_fill,hgrid,col_grid,page_border,ss_border,col_hdr_rects,col_hdr_text,hdr_line,axis_line,row_ticks,row_labels,box_rects,box_sub,box_num).configure_view(stroke=None)

st.set_page_config(page_title="TNLAP Instance Explorer", layout="wide")
st.title("TNLAP Instance Explorer")

with st.sidebar:
    st.header("Instanz laden")
    up = st.file_uploader("JSON-Instanz:", type=["json"])
    if up is None:
        st.info("Lade eine JSON-Datei hoch.")
        st.stop()
    instance = json.load(up)
    st.success("Instanz geladen ✅")
    inst_sig = (getattr(up,"name",None), getattr(up,"size",None))
    if st.session_state.get("_instance_sig") != inst_sig:
        st.session_state["_instance_sig"] = inst_sig
        st.session_state.pop("_shell_to_articles", None)
    W, H = canvas_wh(instance)
    st.caption(f"Grid-Ausdehnung: **{W:.0f} x {H:.0f}** Einheiten")
    n_pages=len(instance.get("pages",[])); n_layouts=len(instance.get("layouts",[]))
    n_articles=len(instance.get("article",[])); n_shells=len(instance.get("shells",[]))
    n_sections=len(instance.get("sections",[]))
    st.markdown(f"**Seiten:** {n_pages} · **Layouts:** {n_layouts} · **Artikel:** {n_articles} · **Shells:** {n_shells} · **Sections:** {n_sections}")
    pages = [int(p) for p in instance.get("pages", [])]
    if not pages: st.error("In der JSON fehlen `pages`."); st.stop()
    st.divider()
    page_id = st.selectbox("Seite wählen", pages, index=0)
    layouts = get_page_layouts(instance, page_id)
    if not layouts: st.warning(f"Keine Layouts für Seite {page_id} gefunden."); st.stop()
    st.write(f"Layouts auf Seite {page_id}: **{len(layouts)}**")

if "chosen_layout" not in st.session_state:
    st.session_state["chosen_layout"] = int(layouts[0])
if "chosen_box" not in st.session_state:
    st.session_state["chosen_box"] = None

tab_explorer, tab_sections, tab_solution = st.tabs(["📐 Layout Explorer", "📋 Sections", "✅ Solution"])

with tab_explorer:
    st.subheader(f"Seite {page_id}: Layout-Vorschau")
    cols = 4
    nrows = math.ceil(len(layouts) / cols)
    for r in range(nrows):
        ccols = st.columns(cols, gap="medium")
        for c in range(cols):
            idx = r*cols+c
            if idx >= len(layouts): continue
            lid = int(layouts[idx])
            with ccols[c]:
                is_active = (st.session_state["chosen_layout"] == lid)
                btn_label = f"✅ Layout {lid}" if is_active else f"Layout {lid}"
                if st.button(btn_label, key=f"pick_layout_{page_id}_{lid}", use_container_width=True):
                    st.session_state["chosen_layout"] = lid
                    st.session_state["chosen_box"] = None
                df_prev = layout_rects_df(instance, lid)
                prev_chart = preview_chart(df_prev, instance, width_px=180, active=is_active)
                if prev_chart is None: st.warning("Leeres Layout.")
                else: st.altair_chart(prev_chart, use_container_width=False)
    st.divider()
    chosen_layout = int(st.session_state["chosen_layout"])
    st.subheader(f"Layout {chosen_layout}: Boxen untersuchen")
    df = layout_rects_df(instance, chosen_layout)
    main_chart = box_chart(df, instance, width_px=460)
    if main_chart is None: st.error("Keine Box-Daten für dieses Layout."); st.stop()
    st.altair_chart(main_chart, use_container_width=False)
    _, H_grid = canvas_wh(instance)
    ROW_SCALE_ui = 100/H_grid
    tbl = df[["box","num_shells","w","h","character"]].copy()
    tbl["Col. von"] = (df["x0"]*1).round().astype(int)+1
    tbl["Col. bis"] = (df["x1"]*1).round().astype(int)
    tbl["Zeile von"] = (df["y0"]*ROW_SCALE_ui).round().astype(int)+1
    tbl["Zeile bis"] = (df["y1"]*ROW_SCALE_ui).round().astype(int)
    tbl["Breite (Col.)"] = (df["w"]).round().astype(int)
    tbl["Höhe (R.)"] = (df["h"]*ROW_SCALE_ui).round().astype(int)
    box_table = tbl[["box","Col. von","Col. bis","Breite (Col.)","Zeile von","Zeile bis","Höhe (R.)","num_shells","character"]].sort_values("box").reset_index(drop=True)
    box_table = box_table.rename(columns={"character":"capacity (chars)","num_shells":"#shells"})
    st.caption("⬇️ Box-Auswahl")
    st.dataframe(box_table, use_container_width=True, hide_index=True)
    box_list = box_table["box"].tolist()
    if not box_list: st.warning("Keine Boxen in diesem Layout."); st.stop()
    default_idx = 0
    if st.session_state.get("chosen_box") in box_list:
        default_idx = box_list.index(st.session_state["chosen_box"])
    chosen_box = st.selectbox("Box auswählen", box_list, index=default_idx, key="chosen_box_selectbox")
    st.session_state["chosen_box"] = int(chosen_box)
    colA, colB = st.columns([1, 1.3], gap="large")
    with colA:
        st.markdown("### Box-Details")
        st.write(f"**Box:** {chosen_box}")
        geom = get_box_geometry(instance, chosen_layout, chosen_box)
        if geom:
            st.write(f"**Geometrie (Grid):** x={geom['x']:.0f}, y={geom['y']:.0f}, w={geom['w']:.0f}, h={geom['h']:.0f}")
            st.write(f"**Zeichenkapazität:** {int(geom['character'])} chars")
        hs = shells_for_layout_box(instance, chosen_layout, chosen_box)
        st.write(f"**Shells in dieser Box:** {hs if hs else '—'}")
        if hs:
            filter_on = st.session_state.get("filter_on", False)
            under_thr = st.session_state.get("under_thr", 15.0)
            over_thr  = st.session_state.get("over_thr",  20.0)
            allowed_arts = set()
            for sec in get_sections_for_page(instance, page_id):
                allowed_arts.update(get_articles_for_section(instance, sec))
            shell_rows = []
            for h in hs:
                sp = shell_params(instance, h)
                hmin = int(sp.get("min", 0)); hmax = int(sp.get("max", 0))
                arts_h = [a for a in compatible_articles_for_shell(instance, h) if a in allowed_arts]
                if filter_on:
                    def _passes(a):
                        L = article_len(instance, a)
                        u = 100.0*(hmin-L)/hmin if L<hmin and hmin>0 else 0.0
                        o = 100.0*(L-hmax)/hmax if L>hmax and hmax>0 else 0.0
                        return u<=under_thr and o<=over_thr
                    arts_h = [a for a in arts_h if _passes(a)]
                shell_rows.append({"shell":h,"min":hmin,"max":hmax,"#articles":len(arts_h)})
            st.dataframe(pd.DataFrame(shell_rows).sort_values("shell"), use_container_width=True, hide_index=True)
    with colB:
        st.markdown("### Shell untersuchen")
        hs = shells_for_layout_box(instance, chosen_layout, chosen_box)
        if not hs:
            st.warning("Keine Shells für diese Box.")
        else:
            shell_id = st.selectbox("Shell wählen", hs, index=0, key="chosen_shell_select")
            sp = shell_params(instance, shell_id)
            smin = int(sp.get("min",0)); smax = int(sp.get("max",0))
            st.write(f"**Shell {shell_id}**")
            st.write(f"- min chars: **{smin}**")
            st.write(f"- max chars: **{smax}**")
            st.markdown("#### 🔎 Filter: Under-/Overfill-Threshold")
            filter_on = st.checkbox("Filter aktivieren (Artikel mit zu starkem Under/Overfill ausblenden)", value=False, key="filter_on")
            c1, c2 = st.columns(2)
            with c1:
                under_thr = st.slider("Max. Underfill (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, disabled=not filter_on, key="under_thr")
            with c2:
                over_thr = st.slider("Max. Overfill (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, disabled=not filter_on, key="over_thr")
            arts = compatible_articles_for_shell(instance, shell_id)
            allowed_arts = set()
            for sec in get_sections_for_page(instance, page_id):
                allowed_arts.update(get_articles_for_section(instance, sec))
            arts = [a for a in arts if a in allowed_arts]
            if not arts:
                st.warning("Keine kompatiblen Artikel für diesen Shell gefunden.")
            else:
                rows = []
                for a in arts:
                    L = article_len(instance, a); pr = article_prio(instance, a)
                    secs = get_pages_for_article(instance, a)
                    under_pct = 100.0*(smin-L)/smin if smin>0 and L<smin else 0.0
                    over_pct  = 100.0*(L-smax)/smax if smax>0 and L>smax else 0.0
                    fits = (L>=smin) and (L<=smax)
                    rows.append({"article":int(a),"prio":pr,"length":int(L),"sections":", ".join(map(str,secs)),"fits":fits,"underfill_%":round(under_pct,2),"overfill_%":round(over_pct,2)})
                adf = pd.DataFrame(rows)
                if filter_on:
                    before = len(adf)
                    adf = adf[(adf["underfill_%"]<=under_thr)&(adf["overfill_%"]<=over_thr)]
                    st.caption(f"Filter aktiv: {before-len(adf)} ausgeblendet, {len(adf)} übrig.")
                if adf.empty:
                    st.warning("Keine Artikel nach Filter.")
                else:
                    prio_order = {"A":0,"B":1,"C":2}
                    adf["_p"] = adf["prio"].map(lambda p: prio_order.get(p,9))
                    adf["misfit_%"] = adf[["underfill_%","overfill_%"]].max(axis=1)
                    adf = adf.sort_values(["fits","_p","misfit_%","length"],ascending=[False,True,True,True]).drop(columns=["_p"])
                    adf["fits"] = adf["fits"].map(lambda x: "✅" if x else "❌")
                    st.dataframe(adf, use_container_width=True, hide_index=True)
    st.divider()

with tab_sections:
    st.subheader("📋 Sections")
    for pid in [int(p) for p in instance.get("pages", [])]:
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
                    st.dataframe(pd.DataFrame([{"artikel":a,"prio":article_prio(instance,a),"länge":article_len(instance,a)} for a in arts]),
                                 use_container_width=True, hide_index=True)
        st.divider()

with tab_solution:
    st.subheader("✅ Solution Viewer")
    sol_file = st.file_uploader("Solution-Datei (.sol)", type=["sol"], key="sol_upload")
    if sol_file is None:
        st.info("Lade eine .sol-Datei hoch.")
    else:
        sol: Dict[str, float] = {}
        objective = None
        for line in sol_file.read().decode("utf-8").splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith("# Objective"):
                try: objective = float(line.split("=")[1].strip())
                except: pass
                continue
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) == 2:
                sol[parts[0]] = float(parts[1])
        page_layout: Dict[int, int] = {}
        for k, v in sol.items():
            if k.startswith("y_") and v == 1:
                p = k.split("_"); page_layout[int(p[1])] = int(p[2])
        assignments: List[Dict] = []
        for k, v in sol.items():
            if k.startswith("x_") and v == 1:
                p = k.split("_")
                assignments.append({"art":int(p[1]),"page":int(p[2]),"layout":int(p[3]),"box":int(p[4]),"shell":int(p[5])})
        pages_sol = sorted(page_layout.keys())
        h1, h2, h3 = st.columns(3)
        h1.metric("Objective", f"{objective:.4f}" if objective is not None else "—")
        h2.metric("Platzierte Artikel", len(assignments))
        h3.metric("Seiten in Solution", len(page_layout))
        st.divider()

        def fill_color(b, pid, layout_id, box_to_art, sol, instance):
            if b not in box_to_art: return "#dc2626"
            e_ov = sol.get(f"e_over_{pid}_{layout_id}_{b}", 0)
            e_un = sol.get(f"e_under_{pid}_{layout_id}_{b}", 0)
            if e_ov or e_un: return "#dc2626"
            d_ov = sol.get(f"delta_over_{pid}_{layout_id}_{b}", 0)
            d_un = sol.get(f"delta_under_{pid}_{layout_id}_{b}", 0)
            if d_ov or d_un:
                a  = box_to_art[b]
                sp = shell_params(instance, a["shell"])
                smin = int(sp.get("min", 0)); smax = int(sp.get("max", 0))
                L = article_len(instance, a["art"])
                if d_ov and smax > 0:
                    t = min(max((L - smax) / smax / 0.3, 0.0), 1.0)
                    r = int(219 + (30  - 219) * t); g = int(234 + (64  - 234) * t); c = int(254 + (175 - 254) * t)
                    return f"#{r:02x}{g:02x}{c:02x}"
                if d_un and smin > 0:
                    t = min(max((smin - L) / smin / 0.3, 0.0), 1.0)
                    r = int(237 + (76  - 237) * t); g = int(233 + (29  - 233) * t); c = int(254 + (149 - 254) * t)
                    return f"#{r:02x}{g:02x}{c:02x}"
            return "#86efac"

        for pid in pages_sol:
            layout_id        = page_layout[pid]
            page_assignments = [a for a in assignments if a["page"] == pid]
            assigned_boxes   = {a["box"] for a in page_assignments}
            all_boxes        = get_layout_boxes(instance, layout_id)
            n_filled = len(assigned_boxes); n_total = len(all_boxes)
            f_page   = sol.get(f"f_page_{pid}")
            fill_str = f"{f_page:.1%}" if f_page is not None else "—"

            with st.expander(f"Seite {pid}  ·  Layout {layout_id}  ·  {n_filled}/{n_total} Boxen belegt", expanded=True):
                col_legend, col_chart, col_table = st.columns([0.55, 1, 1.5], gap="large")

                # ── Legende — Schrift weiß
                with col_legend:
                    st.markdown("**Legende**")
                    def _grad_css(c1, c2): return f"linear-gradient(to right, {c1}, {c2})"
                    def _swatch(color, label):
                        return (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                                f'<span style="display:inline-block;width:16px;height:16px;min-width:16px;'
                                f'background:{color};border:1px solid #55555540;border-radius:3px"></span>'
                                f'<span style="font-size:12px;color:#ffffff">{label}</span></div>')
                    def _grad_row(c1, c2, label):
                        return (f'<div style="margin:4px 0">'
                                f'<div style="font-size:11px;color:#ffffff;margin-bottom:2px">{label}</div>'
                                f'<div style="display:flex;align-items:center;gap:6px">'
                                f'<div style="width:80px;height:14px;border-radius:3px;border:1px solid #55555530;'
                                f'background:{_grad_css(c1,c2)}"></div>'
                                f'</div></div>')
                    html = (
                        _swatch("#02A83F", "Passt") +
                        "<div style='height:8px'></div>" +
                        _swatch("#dc2626", "Leer") +
                        _grad_row("#70aaf5", "#1e40af", "Overfill") +
                        _grad_row("#937df9", "#4c1d95", "Underfill") +
                        "<div style='height:8px'></div>"
                    )
                    st.markdown(f'<div style="padding:10px 0">{html}</div>', unsafe_allow_html=True)

                # ── Chart
                with col_chart:
                    df_layout = layout_rects_df(instance, layout_id)
                    if not df_layout.empty:
                        box_to_art = {a["box"]: a for a in page_assignments}
                        def _ovun(b):
                            if b not in box_to_art: return 0.0, 0.0
                            a = box_to_art[b]
                            sp = shell_params(instance, a["shell"])
                            smin = int(sp.get("min", 0)); smax = int(sp.get("max", 0))
                            L = article_len(instance, a["art"])
                            ov = round(100*(L-smax)/smax, 2) if smax>0 and L>smax else 0.0
                            un = round(100*(smin-L)/smin, 2) if smin>0 and L<smin else 0.0
                            return ov, un

                        df_layout["color"]   = df_layout["box"].apply(lambda b: fill_color(b, pid, layout_id, box_to_art, sol, instance))
                        df_layout["artikel"] = df_layout["box"].apply(lambda b: f"A: {box_to_art[b]['art']}" if b in box_to_art else "leer")
                        df_layout["prio"]    = df_layout["box"].apply(lambda b: article_prio(instance, box_to_art[b]["art"]) if b in box_to_art else "—")
                        df_layout["shell_s"] = df_layout["box"].apply(lambda b: str(box_to_art[b]["shell"]) if b in box_to_art else "—")
                        df_layout["länge"]   = df_layout["box"].apply(lambda b: article_len(instance, box_to_art[b]["art"]) if b in box_to_art else 0)
                        def _ovun_label(b):
                            if b not in box_to_art: return ""
                            ov, un = _ovun(b)
                            if ov > 0: return f"+{ov:.2f}%"
                            if un > 0: return f"-{un:.2f}%"
                            return ""
                        df_layout["ovun_lbl"] = df_layout["box"].apply(_ovun_label)

                        W, H = canvas_wh(instance)
                        FW = W + 2*MARGIN_X; FH = H + 2*MARGIN_Y
                        width_px = 260
                        NO_AXIS = _no_axis()
                        xs = alt.Scale(domain=[0, FW], nice=False)
                        ys = alt.Scale(domain=[0, FH], nice=False, reverse=True)
                        sdf = _shift_df(df_layout)
                        sdf["plx"]    = (sdf["x0"]+sdf["x1"])/2
                        sdf["ply"]    = (sdf["y0"]+sdf["y1"])/2
                        sdf["p_art"]  = (sdf["y0"]*0.6 + sdf["y1"]*0.4)
                        sdf["p_prio"] = (sdf["y0"]*0.4 + sdf["y1"]*0.6)
                        sdf["p_ovun"] = sdf["y1"] - 0.15
                        base_layers, _, _, height_px = _newspaper_layers(instance, width_px)
                        def props(): return dict(width=width_px, height=height_px)

                        rects = alt.Chart(sdf).mark_rect(stroke="#1e293b", strokeWidth=1.1).encode(
                            x=alt.X("x0:Q", axis=NO_AXIS, scale=xs), x2="x1:Q",
                            y=alt.Y("y0:Q", axis=NO_AXIS, scale=ys), y2="y1:Q",
                            color=alt.Color("color:N", scale=None, legend=None),
                            tooltip=[alt.Tooltip("box:N",title="Box"),alt.Tooltip("shell_s:N",title="Shell"),
                                     alt.Tooltip("artikel:N",title="Artikel"),alt.Tooltip("prio:N",title="Prio"),
                                     alt.Tooltip("länge:Q",title="Länge (Zeichen)"),alt.Tooltip("ovun_lbl:N",title="Abweichung")],
                        ).properties(**props())
                        lbl_art = alt.Chart(sdf).mark_text(fontSize=15, fontWeight="bold", color="#1e293b", baseline="bottom", align="center").encode(
                            x=alt.X("plx:Q", axis=NO_AXIS, scale=xs), y=alt.Y("p_art:Q", axis=NO_AXIS, scale=ys), text=alt.Text("artikel:N")).properties(**props())
                        lbl_prio = alt.Chart(sdf).mark_text(fontSize=12, fontWeight="bold", color="#374151", baseline="top", align="center").encode(
                            x=alt.X("plx:Q", axis=NO_AXIS, scale=xs), y=alt.Y("p_prio:Q", axis=NO_AXIS, scale=ys), text=alt.Text("prio:N")).properties(**props())
                        lbl_ovun = alt.Chart(sdf).mark_text(fontSize=11, color="#374151", baseline="bottom", align="center").encode(
                            x=alt.X("plx:Q", axis=NO_AXIS, scale=xs), y=alt.Y("p_ovun:Q", axis=NO_AXIS, scale=ys), text=alt.Text("ovun_lbl:N")).properties(**props())
                        chart = alt.layer(*base_layers, rects, lbl_art, lbl_prio, lbl_ovun)
                        st.altair_chart(chart.configure_view(stroke=None), use_container_width=False)

                # ── Detailtabelle
                with col_table:
                    rows = []
                    for a in sorted(page_assignments, key=lambda x: x["box"]):
                        sp   = shell_params(instance, a["shell"])
                        smin = int(sp.get("min", 0)); smax = int(sp.get("max", 0))
                        L    = article_len(instance, a["art"])
                        fits = smin <= L <= smax
                        under = round(100*(smin-L)/smin, 2) if L<smin and smin>0 else 0.0
                        over  = round(100*(L-smax)/smax, 2) if L>smax and smax>0 else 0.0
                        rows.append({"Box": a["box"], "Shell": a["shell"], "Artikel": a["art"],
                                     "Priorität": article_prio(instance, a["art"]), "Länge": L,
                                     "Shell min": smin, "Shell max": smax,
                                     "Overfill (%)": over, "Underfill (%)": under,
                                     "Fits": "✅" if fits else "❌"})
                    for b in all_boxes:
                        if b not in assigned_boxes:
                            rows.append({"Box": b, "Shell": "—", "Artikel": "—", "Priorität": "—",
                                         "Länge": 0, "Shell min": 0, "Shell max": 0,
                                         "Overfill (%)": 0.0, "Underfill (%)": 0.0, "Fits": "—"})
                    if rows:
                        st.dataframe(pd.DataFrame(rows).sort_values("Box"), use_container_width=True, hide_index=True)