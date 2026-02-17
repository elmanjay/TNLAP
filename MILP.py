from gurobipy import Model, GRB, quicksum
from parser import parse_json_from_file
import os
from analyse import analyse_solution


def create_model(
    pages, article, layouts, resorts, article_resorts, resort_page, layouts_pages,
    box_layouts, hull_layout_box, hull_article, article_length, hull_params,
    article_priority, alpha_value
):
    prioritäten = [1, 0.5, 0.1, 0]
    kategorien = ["A", "B", "C", "D"]
    prio_dict = dict(zip(kategorien, prioritäten))

    g_under_fix = 0.51
    g_under_var = 0.48
    g_over_fix = 0.01
    g_over_var = 0.49
    g_fehlt = 1

    alpha = alpha_value
    T_over = 0.2
    T_under = 0.15
    M_prio = 1


    # ---------------------------
    # Schritt A: Artikel pro Seite (Union über Resorts) einmal vorab berechnen
    # ---------------------------
    articles_on_page = {
        j: set().union(*(set(article_resorts[r]) for r in resort_page[j]))
        for j in pages
    }

    # ---------------------------
    # Längenfilter: hull_article_new
    # ---------------------------
    hull_article_new = {}
    for i in article:
        hull_article_new[i] = []
        for hull in hull_article[i]:
            if (hull_params[hull]["min"] * (1 - T_under)
                <= article_length[i]
                <= hull_params[hull]["max"] * (1 + T_over)):
                hull_article_new[i].append(hull)

    # Big-M (wie bei dir)
    max_hull_value = max(h["max"] for h in hull_params.values())
    max_article_length = max(article_length.values())
    M = max(max_hull_value, max_article_length)
    M_prio = 1

    # ---------------------------
    # Schritt B: hull_layout_box_article mit hull_article_new
    # ---------------------------
    hull_layout_box_article = {}
    for l in layouts:
        hull_layout_box_article[l] = {}
        for k in hull_layout_box[l]:
            hull_layout_box_article[l][k] = {}
            for i in article:
                common = set(hull_layout_box[l][k]) & set(hull_article[i])
                hull_layout_box_article[l][k][i] = list(common)  # ggf. []

    model = Model("milp_model")

    # ---------------------------
    # Schritt C: x nur für zulässige Keys erzeugen (tuplelist -> addVars)
    # ---------------------------
    x_keys = []
    for j in pages:
        Aj = articles_on_page[j]
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                for i in Aj:
                    for h in hull_layout_box_article[l][k].get(i, []):
                        x_keys.append((i, j, l, k, h))

    x = {}
    for (i, j, l, k, h) in x_keys:
        x[i, j, l, k, h] = model.addVar(
            vtype=GRB.BINARY,
            name=f"x_{i}_{j}_{l}_{k}_{h}"
        )

    # ---------------------------
    # Schritt D: y und z ebenfalls kompakt erzeugen (optional, aber sauber)
    # ---------------------------
    y = {}
    y_keys = [(j, l) for j in pages for l in layouts_pages[j]]
    for (j, l) in y_keys:
        y[j, l] = model.addVar(
            vtype=GRB.BINARY,
            name=f"y_{j}_{l}"
        )

    z = {}
    z_keys = [
        (j, l, k, h)
        for j in pages
        for l in layouts_pages[j]
        for k in box_layouts[l]
        for h in hull_layout_box[l][k]
    ]
    for (j, l, k, h) in z_keys:
        z[j, l, k, h] = model.addVar(
            vtype=GRB.BINARY,
            name=f"z_{j}_{l}_{k}_{h}"
        )
    # ---------------------------------------------------------------------
    # Ab hier: dein restlicher Code (Variablen + Constraints) unverändert,
    #          nur kleine Fixes an constraint names, damit es nicht crasht.
    # ---------------------------------------------------------------------

    min_hull = {}
    max_hull = {}
    c_box = {}
    p_box = {}
    v = {}
    delta_over = {}
    delta_under = {}
    e_over = {}
    e_under = {}
    c_box_under = {}
    c_box_over = {}
    f_box = {}

    for j in pages:
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                c_box[j, l, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"c_box_{j}_{l}_{k}")
                f_box[j, l, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"f_box_{j}_{l}_{k}")
                p_box[j, l, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"p_box_{j}_{l}_{k}")
                v[j, l, k] = model.addVar(vtype=GRB.BINARY, name=f"v_{j}_{l}_{k}")
                delta_over[j, l, k] = model.addVar(vtype=GRB.BINARY, name=f"delta_over_{j}_{l}_{k}")
                delta_under[j, l, k] = model.addVar(vtype=GRB.BINARY, name=f"delta_under_{j}_{l}_{k}")
                e_over[j, l, k] = model.addVar(vtype=GRB.BINARY, name=f"e_over_{j}_{l}_{k}")
                e_under[j, l, k] = model.addVar(vtype=GRB.BINARY, name=f"e_under_{j}_{l}_{k}")
                c_box_under[j, l, k] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"c_box_under_{j}_{l}_{k}")
                c_box_over[j, l, k] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"c_box_over_{j}_{l}_{k}")

    for j in pages:
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                hull_length_min = [hull_params[h]["min"] for h in hull_layout_box[l][k]]
                hull_length_max = [hull_params[h]["max"] for h in hull_layout_box[l][k]]
                if len(hull_length_min) > 1:
                    min_hull[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max(hull_length_min), name=f"min_hull_{j}_{l}_{k}")
                    max_hull[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max(hull_length_max), name=f"max_hull_{j}_{l}_{k}")
                else:
                    min_hull[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"min_hull_{j}_{l}_{k}")
                    max_hull[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"max_hull_{j}_{l}_{k}")

    c_page = {j: model.addVar(vtype=GRB.CONTINUOUS, name=f"c_page_{j}") for j in pages}
    p_page = {j: model.addVar(vtype=GRB.CONTINUOUS, name=f"p_page_{j}") for j in pages}
    f_page = {j: model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"f_page_{j}") for j in pages}

    p_page_layout = {}
    for j in pages:
        for l in layouts_pages[j]:
            p_page_layout[j, l] = model.addVar(vtype=GRB.CONTINUOUS, name=f"p_page_layout_{j}_{l}")

    objective = quicksum(f_page[j] for j in pages)
    model.setObjective(objective, GRB.MAXIMIZE)

    # NB2
    model.addConstrs(
        (quicksum(y[j, l] for l in layouts_pages[j]) == 1 for j in pages),
        name="NB2"
    )

    # NB3
    model.addConstrs(
        (quicksum(z[j, l, k, h] for h in hull_layout_box[l][k]) == y[j, l]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB3"
    )

    # NB4 
    model.addConstrs(
        (x[i, j, l, k, h] <= z[j, l, k, h] for (i, j, l, k, h) in x_keys),
        name="NB4"
    )


    # NB5
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h]
            for j in pages
            for l in layouts_pages[j]
            for k in box_layouts[l]
            if any(i in article_resorts[r] for r in resort_page[j])
            for h in hull_layout_box_article[l][k][i]
        ) <= 1 for i in article),
        name="NB5"
    )

    # NB6
    model.addConstrs(
        (min_hull[j, l, k] == quicksum(hull_params[h]["min"] * z[j, l, k, h] for h in hull_layout_box[l][k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB6"
    )

    # NB7
    model.addConstrs(
        (max_hull[j, l, k] == quicksum(hull_params[h]["max"] * z[j, l, k, h] for h in hull_layout_box[l][k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB7"
    )

    # NB8
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) == y[j, l] - v[j, l, k]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB8"
    )

    # NB9a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) >= min_hull[j, l, k] - M * (delta_under[j, l, k] + v[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB9a"
    )

    # NB9b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) <= min_hull[j, l, k] + M * (1 - delta_under[j, l, k] - v[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB9b"
    )

    # NB10a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) <= max_hull[j, l, k] + M * delta_over[j, l, k]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB10a"
    )

    # NB10b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) >= max_hull[j, l, k] - M * (1 - delta_over[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB10b"
    )

    # NB11a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) <= (1 + T_over) * max_hull[j, l, k] + M * e_over[j, l, k]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB11a"
    )

    # NB11b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) >= (1 + T_over) * max_hull[j, l, k] - M * (1 - e_over[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB11b"
    )

    # NB12a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) <= (1 - T_under) * min_hull[j, l, k] + M * (1 - e_under[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB12a"
    )

    # NB12b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) >= (1 - T_under) * min_hull[j, l, k] - M * (e_under[j, l, k] + v[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB12b"
    )

    # NB13
    model.addConstrs(
        (p_box[j, l, k] <= quicksum(
            x[i, j, l, k, h] * prio_dict[article_priority[i]]
            for r in resort_page[j]
            for i in article_resorts[r]
            for h in hull_layout_box_article[l][k][i]
        ) 
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB13"
    )

    # NB14
    model.addConstrs(
        (c_box_under[j, l, k] >=
         g_under_fix + g_under_var *
         quicksum(
             x[i, j, l, k, h] * (article_length[i] / hull_params[h]["min"] - 1)
             for r in resort_page[j]
             for i in article_resorts[r]
             for h in hull_layout_box_article[l][k][i]
         ) / T_under * -1
         - ((1 - delta_under[j, l, k]) + e_under[j, l, k]) * M_prio
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB14a"
    )

    model.addConstrs(
        (c_box_under[j, l, k] <=
         g_under_fix + g_under_var *
         quicksum(
             x[i, j, l, k, h] * (article_length[i] / hull_params[h]["min"] - 1)
             for r in resort_page[j]
             for i in article_resorts[r]
             for h in hull_layout_box_article[l][k][i]
         ) / T_under * -1
         + ((1 - delta_under[j, l, k]) + e_under[j, l, k]) * M_prio
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB14b"
    )

    # NB15
    model.addConstrs(
        (c_box_over[j, l, k] >=
         g_over_fix + g_over_var *
         quicksum(
             x[i, j, l, k, h] * (article_length[i] / hull_params[h]["max"] - 1)
             for r in resort_page[j]
             for i in article_resorts[r]
             for h in hull_layout_box_article[l][k][i]
         ) / T_over
         - ((1 - delta_over[j, l, k]) + e_over[j, l, k]) * M_prio
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB15"
    )

    model.addConstrs(
        (c_box_over[j, l, k] <=
         g_over_fix + g_over_var *
         quicksum(
             x[i, j, l, k, h] * (article_length[i] / hull_params[h]["max"] - 1)
             for r in resort_page[j]
             for i in article_resorts[r]
             for h in hull_layout_box_article[l][k][i]
         ) / T_over
         + ((1 - delta_over[j, l, k]) + e_over[j, l, k]) * M_prio
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB15b"
    )

    # NB16
    model.addConstrs(
        (c_box[j, l, k] == y[j,l]- ( c_box_under[j, l, k] + c_box_over[j, l, k]
         + (e_over[j, l, k] + e_under[j, l, k] + v[j, l, k]) * g_fehlt)
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB16"
    )

    # NB17
    model.addConstrs(
        (f_box[j, l, k] == alpha * p_box[j, l, k] + (1 - alpha) * c_box[j, l, k]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB17"
    )

    # NB18
    model.addConstrs(
        (f_page[j] == quicksum(
            1 / len(box_layouts[l]) * f_box[j, l, k]
            for l in layouts_pages[j]
            for k in box_layouts[l]
        ) for j in pages),
        name="NB18"
    )

    # NB19
    model.addConstrs(
        (delta_over[j, l, k] <= y[j, l]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB19"
    )

    return model


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_name = "H50P250A1(A)" 
    instance_dir = os.path.join(base_dir, "Instances")
    name = os.path.join(instance_dir, f"{instance_name}.json")

    # --- Verzeichnis-Struktur sicherstellen ---
    lp_dir = os.path.join(instance_dir, "lp")
    lp_mps_dir = os.path.join(lp_dir, "mps")
    sol_dir = os.path.join(instance_dir, "sol")

    os.makedirs(lp_dir, exist_ok=True)
    os.makedirs(lp_mps_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)
    # ------------------------------------------

    alpha_value = 0.1
    canvas, pages, article, layouts, resorts, article_resorts, resort_page, layouts_pages, box_layouts, box_geomtery, hull_layout_box, hull_article, article_length, hull_params, article_priority = parse_json_from_file(name)

    model = create_model(
        pages, article, layouts, resorts, article_resorts, resort_page, layouts_pages,
        box_layouts, hull_layout_box, hull_article, article_length, hull_params,
        article_priority, alpha_value
    )

    model.setParam('TimeLimit', 3600)
    model.Params.LogFile = os.path.join(sol_dir, f"{instance_name}.log")
    model.Params.Threads = 1

    model.write(os.path.join(lp_mps_dir, f"{instance_name}.mps"))

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Modell ist unlösbar. Berechne IIS...")
        model.computeIIS()
        model.write("model.ilp")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"IIS enthält Constraint: {c.ConstrName}")
    else:
        model.write(os.path.join(lp_dir, f"{instance_name}.lp"))
        model.write(os.path.join(sol_dir, f"{instance_name}.sol"))
        analyse_solution(model, article_length, hull_params, article_priority)



    
