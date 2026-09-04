from gurobipy import Model, GRB, quicksum
from parser import parse_json_from_file
import os
#import analyse


# =====================================================================
# TNLAP - kollabierte Formulierung
# ---------------------------------------------------------------------
# Voraussetzung: hoechstens ein Artikel pro Box (wie NB8 im Originalmodell).
# Dann haengt die Fallklassifikation (EXACT/UNDER/OVER/UNDER_VIOL/
# OVER_VIOL) und damit die Strafe NUR vom Paar (Artikel i, Shell h) ab
# und ist eine Konstante -> Preprocessing statt Big-M/Indikatoren.
#
#   w[i,h] = alpha * prio(i) + (1 - alpha) * (1 - pen(i,h))
#
# Leere Box traegt bei g_fehlt = 1 exakt 0 zur Zielfunktion bei und
# faellt daher aus der Zielfunktion heraus.
#
# Variablen:   x[i,j,l,k,h], y[j,l], z[j,l,k,h]   (alle binaer)
# Zielfunktion: max sum (1/|B_l|) * w[i,h] * x[i,j,l,k,h]
# Constraints: NB2 (ein Layout je Seite), NB3 (eine Shell je Box),
#              NB4 (x <= z), NB5 (Artikel hoechstens einmal),
#              NB8' (hoechstens ein Artikel je Box)
# =====================================================================


# ------------------- Parameter (wie im Original) --------------------
PRIO_DICT = {"A": 1, "B": 0.5, "C": 0.1, "D": 0}

G_UNDER_FIX = 0.51
G_UNDER_VAR = 0.48
G_OVER_FIX = 0.01
G_OVER_VAR = 0.49
G_FEHLT = 1

T_OVER = 0.3
T_UNDER = 0.3


def classify_pair(length, mn, mx):
    """Fallklassifikation und Strafe fuer ein Paar (Artikellaenge, Shell).

    Entspricht der Logik des tnlap_evaluator:
      EXACT      : mn <= L <= mx                     -> pen = 0
      UNDER      : (1-T_UNDER)*mn <= L < mn          -> pen = fix + var * reldev/T
      OVER       : mx < L <= (1+T_OVER)*mx           -> pen = fix + var * reldev/T
      UNDER_VIOL : L < (1-T_UNDER)*mn                -> pen = G_FEHLT
      OVER_VIOL  : L > (1+T_OVER)*mx                 -> pen = G_FEHLT
    """
    if mn <= length <= mx:
        return "EXACT", 0.0
    if length < mn:
        if mn > 0 and length >= (1 - T_UNDER) * mn:
            return "UNDER", G_UNDER_FIX + G_UNDER_VAR * (1 - length / mn) / T_UNDER
        return "UNDER_VIOL", G_FEHLT
    if mx > 0 and length <= (1 + T_OVER) * mx:
        return "OVER", G_OVER_FIX + G_OVER_VAR * (length / mx - 1) / T_OVER
    return "OVER_VIOL", G_FEHLT


def create_model(
    pages, article, layouts, sections, article_sections, sections_page, layouts_pages,
    box_layouts, shells_layout_box, shells_article, article_length, shell_params,
    article_priority, alpha_value, pre_filter, pref_filter_tolerance, geometry_layout_box, layout_vertical_chains
):
    alpha = alpha_value

    # ---------------------------
    # Artikel pro Seite (Union ueber sections)
    # ---------------------------
    articles_on_page = {
        j: set().union(*(set(article_sections[r]) for r in sections_page[j]))
        for j in pages
    }

    # ---------------------------
    # Optionaler Laengenfilter (wie im Original)
    # ---------------------------
    shells_article_new = {}
    for i in article:
        shells_article_new[i] = []
        for shell in shells_article[i]:
            if (shell_params[shell]["min"] * (1 - (T_UNDER + pref_filter_tolerance))
                <= article_length[i]
                <= shell_params[shell]["max"] * (1 + T_OVER + pref_filter_tolerance)):
                shells_article_new[i].append(shell)

    shells_layout_box_article = {}
    for l in layouts:
        shells_layout_box_article[l] = {}
        for k in shells_layout_box[l]:
            shells_layout_box_article[l][k] = {}
            for i in article:
                common = set(shells_layout_box[l][k]) & set(
                    shells_article_new[i] if pre_filter else shells_article[i]
                )
                shells_layout_box_article[l][k][i] = list(common)

    # ---------------------------
    # Preprocessing: Paar-Bewertung w[i,h] und Fallklasse
    # ---------------------------
    w = {}
    pair_case = {}
    for i in article:
        for h in shells_article[i]:
            case, pen = classify_pair(
                article_length[i],
                shell_params[h]["min"],
                shell_params[h]["max"],
            )
            pair_case[i, h] = case
            w[i, h] = alpha * PRIO_DICT[article_priority[i]] + (1 - alpha) * (1 - pen)
    
    print("hehe")
    print(len(w))

    model = Model("tnlap_collapsed")

    # ---------------------------
    # Variablen
    # ---------------------------
    x_keys = []
    for j in pages:
        Aj = articles_on_page[j]
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                for i in Aj:
                    for h in shells_layout_box_article[l][k].get(i, []):
                        x_keys.append((i, j, l, k, h))

    x = model.addVars(x_keys, vtype=GRB.BINARY, name=f"x_{i}_{j}_{l}_{k}_{h}")

    y = model.addVars(
        [(j, l) for j in pages for l in layouts_pages[j]],
        vtype=GRB.BINARY, name=f"y_{j}_{l}"
    )

    z_keys = [
        (j, l, k, h)
        for j in pages
        for l in layouts_pages[j]
        for k in box_layouts[l]
        for h in shells_layout_box[l][k]
    ]
    z = model.addVars(z_keys, vtype=GRB.BINARY, name=f"z_{j}_{l}_{k}_{h}")

    box_index = [
        (j, l, k)
        for j in pages
        for l in layouts_pages[j]
        for k in box_layouts[l]
    ]

    # ---------------------------
    # Zielfunktion
    # ---------------------------
    model.setObjective(
        quicksum(
            (1 / len(box_layouts[l])) * w[i, h] * x[i, j, l, k, h]
            for (i, j, l, k, h) in x_keys
        ),
        GRB.MAXIMIZE
    )


    # NB2: genau ein Layout pro Seite
    model.addConstrs(
        (y.sum(j, "*") == 1 for j in pages),
        name="NB2"
    )

    # NB3: genau eine Shell pro Box, wenn Layout aktiv
    model.addConstrs(
        (quicksum(z[j, l, k, h] for h in shells_layout_box[l][k]) == y[j, l]
         for (j, l, k) in box_index),
        name="NB3"
    )

    # NB4: Zuordnung nur in gewaehlte Shell
    model.addConstrs(
        (x[i, j, l, k, h] <= z[j, l, k, h] for (i, j, l, k, h) in x_keys),
        name="NB4"
    )

    # NB5: jeder Artikel hoechstens einmal
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h]
            for j in pages
            for l in layouts_pages[j]
            for k in box_layouts[l]
            if i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) <= 1 for i in article),
        name="NB5"
    )

    # NB8': hoechstens ein Artikel pro Box
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) <= y[j, l]
         for (j, l, k) in box_index),
        name="NB8"
    )

    return model#, shells_layout_box_article, x, y, z, w, pair_case


def report_solution(model, x, w, pair_case, box_layouts, article_priority):
    """Kompakte Loesungsauswertung auf Basis der Paar-Klassifikation."""
    if model.SolCount == 0:
        print("Keine Loesung gefunden.")
        return
    counts = {}
    placed = 0
    for (i, j, l, k, h), var in x.items():
        if var.X > 0.5:
            placed += 1
            case = pair_case[i, h]
            counts[case] = counts.get(case, 0) + 1
    print(f"\nZielfunktionswert: {model.ObjVal:.6f}")
    print(f"Platzierte Artikel: {placed}")
    for case in ["EXACT", "UNDER", "OVER", "UNDER_VIOL", "OVER_VIOL"]:
        if case in counts:
            print(f"  {case:<11}: {counts[case]}")


if __name__ == "__main__":

    ############ CONFIGURATION ############
    instance_name = "P40_A200_A_118"
    alpha_value = 0.1
    pre_filter = False
    pre_filter_tolerance = 0.6

    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(base_dir, "Instances")
    name = os.path.join(instance_dir, f"{instance_name}.json")
    lp_dir = os.path.join(instance_dir, "lp")
    sol_dir = os.path.join(instance_dir, "sol")

    os.makedirs(lp_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)

    (pages, article, layouts, sections, article_sections, sections_page,
     layouts_pages, box_layouts, box_geomtery, shells_layout_box, shells_article,
     article_length, shell_params, article_priority) = parse_json_from_file(name)

    print("start building model...")
    model, shells_layout_box_article, x, y, z, w, pair_case = create_model(
        pages, article, layouts, sections, article_sections, sections_page,
        layouts_pages, box_layouts, shells_layout_box, shells_article,
        article_length, shell_params, article_priority,
        alpha_value, pre_filter, pre_filter_tolerance
    )
    print("model completed. start optimizing...")
    model.setParam('TimeLimit', 3600)
    model.Params.LogFile = os.path.join(sol_dir, f"{instance_name}_collapsed.log")
    model.Params.Threads = 1
    model.write(os.path.join(lp_dir, f"{instance_name}_collapsed.lp"))

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Modell ist unloesbar. Berechne IIS...")
        model.computeIIS()
        model.write("model.ilp")
    else:
        model.write(os.path.join(sol_dir, f"{instance_name}_collapsed.sol"))
        report_solution(model, x, w, pair_case, box_layouts, article_priority)
        #analyse.analyse_solution(model, article_length, shell_params, article_priority)
