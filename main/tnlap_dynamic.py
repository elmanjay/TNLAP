from gurobipy import Model, GRB, quicksum

def create_model(
    pages, article, layouts, sections, article_sections, sections_page, layouts_pages,
    box_layouts, shells_layout_box, shells_article, article_length, shell_params,
    article_priority, alpha_value, pre_filter, pre_filter_tolerance, geometry_layout_box, layout_vertical_chains
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
    T_over = 0.3
    T_under = 0.3


    # ---------------------------
    # Schritt A: Artikel pro Seite (Union über Resorts) einmal vorab berechnen
    # ---------------------------
    articles_on_page = {
        j: set().union(*(set(article_sections[r]) for r in sections_page[j]))
        for j in pages
    }

    # ---------------------------
    # Längenfilter: hull_article_new, fix für Big M und macht Sinn, würde auch über Instanzgenerierung gehen
    # ---------------------------
    shells_article_new = {}
    for i in article:
        shells_article_new[i] = []
        for shell in shells_article[i]:
            if (shell_params[shell]["min"] * (1 - (T_under + pre_filter_tolerance)) #+ über/unter Schwellwert
                <= article_length[i]
                <= shell_params[shell]["max"] * (1 + T_over + pre_filter_tolerance)):
                shells_article_new[i].append(shell)
    
    # Big-M (wie bei dir)
    max_shell_value = max(h["max"] for h in shell_params.values())
    max_article_length = max(article_length.values())
    M_length = max(max_shell_value, max_article_length)
    """
    M_nb14 = g_under_fix + g_under_var * max(
        abs(article_length[i] / shell_params[h]["min"] - 1)
        for i in article
        for h in shells_article[i]
        if shell_params[h]["min"] > 0
    ) / T_under + 1

    M_nb15 = g_over_fix + g_over_var * max(
        abs(article_length[i] / shell_params[h]["max"] - 1)
        for i in article
        for h in shells_article[i]
        if shell_params[h]["max"] > 0
    ) / T_over + 1
    """
    ###
    # Kleinstmögliche Kapazität einer Box ermitteln (unter Berücksichtigung von h_delta = -5)
    min_possible_shell_min = min(
        max(1, shell_params[h]["min"] - 5 * geometry_layout_box[l][k]["w"] * 40)
        for l in layouts for k in box_layouts[l] for h in shells_layout_box[l][k]
    )
    
    min_possible_shell_max = min(
        max(1, shell_params[h]["max"] - 5 * geometry_layout_box[l][k]["w"] * 40)
        for l in layouts for k in box_layouts[l] for h in shells_layout_box[l][k]
    )

    max_art_len = max(article_length.values())

    # Neue, numerisch stabile M-Werte für die relative Ratio-Berechnung
    M_nb14 = g_under_fix + (g_under_var / T_under) * abs(max_art_len / min_possible_shell_min - 1) + 1
    M_nb15 = g_over_fix + (g_over_var / T_over) * abs(max_art_len / min_possible_shell_max - 1) + 1





    # ---------------------------
    # Schritt B: hull_layout_box_article mit hull_article_new
    # ---------------------------
    shells_layout_box_article = {} #S_i_l_k
    for l in layouts:
        shells_layout_box_article[l] = {}
        for k in shells_layout_box[l]:
            shells_layout_box_article[l][k] = {}
            for i in article:
                common = set(shells_layout_box[l][k]) & set(shells_article_new[i] if pre_filter else shells_article[i])
                shells_layout_box_article[l][k][i] = list(common)  # ggf. []

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
                    for h in shells_layout_box_article[l][k].get(i, []):
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
        for h in shells_layout_box[l][k]
    ]
    for (j, l, k, h) in z_keys:
        z[j, l, k, h] = model.addVar(
            vtype=GRB.BINARY,
            name=f"z_{j}_{l}_{k}_{h}"
        )

    h_delta = {}
    for j in pages:
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                h_delta[j, l, k] = model.addVar(
                    lb=-5.0, ub=5.0, #anpassen auf gewünschte Zeilenverschiebung
                    vtype=GRB.INTEGER,
                    name=f"h_delta_{j}_{l}_{k}"
                )



    # ---------------------------------------------------------------------
    # Ab hier: dein restlicher Code (Variablen + Constraints) unverändert,
    #          nur kleine Fixes an constraint names, damit es nicht crasht.
    # ---------------------------------------------------------------------

    min_shell = {}
    max_shell = {}
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

    ###
    ratio_under = {}
    ratio_over = {}

    for j in pages:
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                # Hilfsvariablen für die exakten Brüche
                ratio_under[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"ratio_under_{j}_{l}_{k}")
                ratio_over[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"ratio_over_{j}_{l}_{k}")

                

    for j in pages:
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                c_box[j, l, k] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"c_box_{j}_{l}_{k}")
                f_box[j, l, k] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"f_box_{j}_{l}_{k}")
                p_box[j, l, k] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"p_box_{j}_{l}_{k}")
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
                shell_length_min = [shell_params[h]["min"] for h in shells_layout_box[l][k]]
                shell_length_max = [shell_params[h]["max"] for h in shells_layout_box[l][k]]
                if len(shell_length_min) > 1:
                    min_shell[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max(shell_length_min), name=f"min_shell_{j}_{l}_{k}")
                    max_shell[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max(shell_length_max), name=f"max_shell_{j}_{l}_{k}")
                else:
                    min_shell[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"min_shell_{j}_{l}_{k}")
                    max_shell[j, l, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"max_shell_{j}_{l}_{k}")

    
    f_page = {j: model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"f_page_{j}") for j in pages}

    

    objective = quicksum(f_page[j] for j in pages)
    model.setObjective(objective, GRB.MAXIMIZE)
    


    # NB2
    model.addConstrs(
        (quicksum(y[j, l] for l in layouts_pages[j]) == 1 for j in pages),
        name="NB2"
    )

    # NB3
    model.addConstrs(
        (quicksum(z[j, l, k, h] for h in shells_layout_box[l][k]) == y[j, l]
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
            if any(i in article_sections[r] for r in sections_page[j])
            #if i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) <= 1 for i in article),
        name="NB5"
    )

    # NB6
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) == y[j, l] - v[j, l, k]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB6"
    )

    # NB 7+8 raus und unten angepasst


    # NB9a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) >= min_shell[j, l, k] - M_length * (delta_under[j, l, k] + v[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB9a"
    )

    # NB9b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) <= min_shell[j, l, k] + M_length * (1 - delta_under[j, l, k] - v[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB9b"
    )

    # NB10a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) <= max_shell[j, l, k] + M_length * delta_over[j, l, k]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB10a"
    )

    # NB10b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) >= max_shell[j, l, k] - M_length * (1 - delta_over[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB10b"
    )

    # NB11a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) <= (1 - T_under) * min_shell[j, l, k] + M_length * (1 - e_under[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB11a"
    )

    # NB11b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) >= (1 - T_under) * min_shell[j, l, k] - M_length * (e_under[j, l, k] + v[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB11b"
    )

    # NB12a
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) <= (1 + T_over) * max_shell[j, l, k] + M_length * e_over[j, l, k]
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB12a"
    )

    # NB12b
    model.addConstrs(
        (quicksum(
            x[i, j, l, k, h] * article_length[i]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) >= (1 + T_over) * max_shell[j, l, k] - M_length * (1 - e_over[j, l, k] + v[j, l, k])
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB12b"
    )

    # NB13
    model.addConstrs(
        (p_box[j, l, k] <= quicksum(
            x[i, j, l, k, h] * prio_dict[article_priority[i]]
            #for r in resort_page[j]
            #for i in article_resorts[r]
            for i in articles_on_page[j]
            for h in shells_layout_box_article[l][k][i]
        ) 
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB13"
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

    # Geometrische Nebenbedingungen:

    # Wenn ein Layout nicht gewählt ist (y=0), muss h_delta 0 sein
    model.addConstrs(
        (h_delta[j, l, k] <= 5 * y[j, l] for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="h_delta_upper"
    )
    model.addConstrs(
        (h_delta[j, l, k] >= -5 * y[j, l] for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="h_delta_lower"
    )

    # Tatsächliche Höhe + h_delta über alle Boxen einer Kette = 100
    for j in pages:
        for l in layouts_pages[j]:
            if l in layout_vertical_chains:
                for idx, chain in enumerate(layout_vertical_chains[l]):
                    model.addConstr(
                        quicksum(
                            h_delta[j, l, k] + geometry_layout_box[l][k]["h"] * y[j, l]
                            for k in chain
                        ) == 100 * y[j, l],
                        name=f"vertical_chain_{j}_{l}_{idx}"
                    )

    # NB 7+8 oben raus und hier in angepasster Version:
    # NB7 (Dynamische Kapazität)
    model.addConstrs(
        (min_shell[j, l, k] == 
         quicksum(shell_params[h]["min"] * z[j, l, k, h] for h in shells_layout_box[l][k])
         + h_delta[j, l, k] * geometry_layout_box[l][k]["w"] * 40
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB7_dynamic"
    )

    # NB8 (Dynamische Kapazität)
    model.addConstrs(
        (max_shell[j, l, k] == 
         quicksum(shell_params[h]["max"] * z[j, l, k, h] for h in shells_layout_box[l][k])
         + h_delta[j, l, k] * geometry_layout_box[l][k]["w"] * 40
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB8_dynamic"
    )

    # Quadratische Nebenbedingung (A * B = C)
    for j in pages:
        for l in layouts_pages[j]:
            for k in box_layouts[l]:
                # Die Summe der zugewiesenen Artikel-Längen
                assigned_length = quicksum(
                    x[i, j, l, k, h] * article_length[i]
                    for i in articles_on_page[j]
                    for h in shells_layout_box_article[l][k][i]
                )
                
                # ratio * dynamische Kapazität = zugewiesene Länge
                model.addConstr(ratio_under[j, l, k] * min_shell[j, l, k] == assigned_length, name=f"quad_under_{j}_{l}_{k}")
                model.addConstr(ratio_over[j, l, k] * max_shell[j, l, k] == assigned_length, name=f"quad_over_{j}_{l}_{k}")

    # NB 14+15 oben raus und hier in angepasster Version:
    # # NB14 (Exakt nicht-linear)
    model.addConstrs(
        (c_box_under[j, l, k] >=
         g_under_fix + g_under_var *
         (ratio_under[j, l, k] - quicksum(
             x[i, j, l, k, h]
             for i in articles_on_page[j]
             for h in shells_layout_box_article[l][k][i]
         )) / T_under * -1
         - ((1 - delta_under[j, l, k]) + e_under[j, l, k]) * M_nb14
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB14a"
    )

    model.addConstrs(
        (c_box_under[j, l, k] <=
         g_under_fix + g_under_var *
         (ratio_under[j, l, k] - quicksum(
             x[i, j, l, k, h]
             for i in articles_on_page[j]
             for h in shells_layout_box_article[l][k][i]
         )) / T_under * -1
         + ((1 - delta_under[j, l, k]) + e_under[j, l, k]) * M_nb14
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB14b"
    )

    # NB15 (Exakt nicht-linear)
    model.addConstrs(
        (c_box_over[j, l, k] >=
         g_over_fix + g_over_var *
         (ratio_over[j, l, k] - quicksum(
             x[i, j, l, k, h]
             for i in articles_on_page[j]
             for h in shells_layout_box_article[l][k][i]
         )) / T_over
         - ((1 - delta_over[j, l, k]) + e_over[j, l, k]) * M_nb15
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB15a"
    )

    model.addConstrs(
        (c_box_over[j, l, k] <=
         g_over_fix + g_over_var *
         (ratio_over[j, l, k] - quicksum(
             x[i, j, l, k, h]
             for i in articles_on_page[j]
             for h in shells_layout_box_article[l][k][i]
         )) / T_over
         + ((1 - delta_over[j, l, k]) + e_over[j, l, k]) * M_nb15
         for j in pages for l in layouts_pages[j] for k in box_layouts[l]),
        name="NB15b"
    )   

    return model        
