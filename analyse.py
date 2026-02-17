from collections import Counter, defaultdict
from pprint import pprint


def _pct(part, total, digits=2):
    return round((part / total) * 100, digits) if total else 0.0


def _h(title, ch="="):
    print(ch * 60)
    print(title)
    print(ch * 60)


def _box_under_over(total_len, hull_min, hull_max):
    """
    Returns a short string like:
      'UNDER -12.3%'  or  'OVER +8.7%'
    or None if OK / not computable.
    """
    if hull_min is None or hull_max is None:
        return None

    if total_len < hull_min:
        pct = round(100 * (total_len - hull_min) / hull_min, 2) if hull_min else 0.0
        return f"UNDER {pct}%"

    if total_len > hull_max:
        pct = round(100 * (total_len - hull_max) / hull_max, 2) if hull_max else 0.0
        return f"OVER +{pct}%"

    return None


def analyse_solution(model, article_length, hull_params, article_priority):
    tol = 1e-9
    print_nonzero_vars = True

    solution_dict = {}
    z_count = v_count = x_count = 0

    placed_articles = []
    placed_priorities = []

    # page -> box -> {art: priority}  (oder "<EMPTY>":0)
    articles_pages = defaultdict(lambda: defaultdict(dict))

    underfill_penalty_sum = 0.0
    overfill_penalty_sum = 0.0
    page_scores = {}

    # page -> chosen layout
    chosen_layout = {}

    # (page, box) -> chosen shell/hull (int) OR list[int] if inconsistent
    chosen_shell = {}  # key: (page, box), value: hull or [hull,...]

    # -------------------------
    # 1) Variablen lesen
    # -------------------------
    for v in model.getVars():
        val = v.X
        if abs(val) <= tol:
            continue

        name = v.VarName
        if print_nonzero_vars:
            print(f"{name}: {val}")

        # Layoutwahl merken: y_<page>_<layout>
        if name.startswith("y_"):
            parts = name.split("_")
            if len(parts) >= 3 and abs(val - 1.0) <= tol:
                page = int(parts[1])
                layout = int(parts[2])
                chosen_layout[page] = layout

        # Shell/Hull-Wahl merken: z_<page>_<layout>_<box>_<hull>
        if name.startswith("z_"):
            parts = name.split("_")
            # Erwartet: z_<page>_<layout>_<box>_<hull>
            if len(parts) >= 5 and abs(val - 1.0) <= tol:
                page = int(parts[1])
                box = int(parts[3])
                hull = int(parts[4])
                key = (page, box)

                # falls aus irgendeinem Grund mehrere z=1 pro Box existieren:
                if key not in chosen_shell:
                    chosen_shell[key] = hull
                else:
                    # schon gesetzt -> liste daraus machen
                    if isinstance(chosen_shell[key], list):
                        chosen_shell[key].append(hull)
                    else:
                        chosen_shell[key] = [chosen_shell[key], hull]

        if name.startswith("z"):
            z_count += 1

        elif name.startswith("v"):
            v_count += 1
            if abs(val - 1.0) <= tol:
                parts = name.split("_")
                # v_<page>_<layout>_<box>
                page = int(parts[1])
                box = int(parts[3])
                articles_pages[page][box]["<EMPTY>"] = 0

        elif name.startswith("x"):
            x_count += 1
            parts = name.split("_")
            # x_<art>_<page>_<layout>_<box>_<hull>
            art = int(parts[1])
            hull = int(parts[-1])
            solution_dict[art] = hull

            if abs(val - 1.0) <= tol:
                page = int(parts[2])
                box = int(parts[4])
                placed_articles.append(art)
                placed_priorities.append(article_priority[art])
                articles_pages[page][box][art] = article_priority[art]

        if name.startswith("c_box_under"):
            underfill_penalty_sum += val
        elif name.startswith("c_box_over"):
            overfill_penalty_sum += val

        if name.startswith("f_page") and len(name.split("_")) >= 3 and name.split("_")[2] != "layout":
            # f_page_<page>
            parts = name.split("_")
            page = int(parts[2])
            page_scores[page] = round(val, 4)

    # -------------------------
    # 2) Under/Overfill Analyse (Artikel vs. Hull)
    # -------------------------
    under_n = over_n = 0
    under_sum = over_sum = 0.0

    under_details = []
    over_details = []

    for art, hull in solution_dict.items():
        art_len = article_length[art]
        hull_min = hull_params[hull]["min"]
        hull_max = hull_params[hull]["max"]

        if art_len < hull_min:
            rate = round(-100 * (art_len - hull_min) / hull_min, 3)
            under_n += 1
            under_sum += rate
            under_details.append((art, hull, rate))

        if art_len > hull_max:
            rate = round(100 * (art_len - hull_max) / hull_max, 3)
            over_n += 1
            over_sum += rate
            over_details.append((art, hull, rate))

    # -------------------------
    # 3) Ausgabe
    # -------------------------
    _h("LÖSUNGSÜBERSICHT")
    print(f"Anzahl ausgewählter Hüllen (z): {z_count}")
    print(f"Anzahl leerer Boxen (v):        {v_count}")
    print(f"Anzahl x-Variablen (x):         {x_count}")

    _h("FÜLLGRAD-ANALYSE")
    print(f"Unterfüllte Artikel: {under_n} ({_pct(under_n, x_count)}%) | Summe: {round(under_sum,3)}%")
    print(f"Überfüllte Artikel:  {over_n} ({_pct(over_n, x_count)}%) | Summe: {round(over_sum,3)}%")
    print(f"Penalty Unterfüllung: {underfill_penalty_sum}")
    print(f"Penalty Überfüllung:  {overfill_penalty_sum}")

    if under_details:
        print("\nUnterfüllung (art, hull, %):")
        for art, hull, rate in sorted(under_details, key=lambda t: -t[2]):
            print(f"  - Artikel {art:>4} | Hülle {hull:>3} | {rate:>8}%")

    if over_details:
        print("\nÜberfüllung (art, hull, %):")
        for art, hull, rate in sorted(over_details, key=lambda t: -t[2]):
            print(f"  - Artikel {art:>4} | Hülle {hull:>3} | {rate:>8}%")

    _h("PRIORITÄTEN")
    print(f"Verteilung in Instanz: {Counter(article_priority.values())}")
    if placed_priorities:
        total = len(placed_priorities)
        for p in ["A", "B", "C"]:
            c = placed_priorities.count(p)
            print(f"{p} platziert: {c:>4} ({_pct(c, total)}%)")
    else:
        print("Keine platzierten Artikel gefunden.")

    _h("SEITENBELEGUNG (LESBAR)")

    pages_to_print = sorted(set(articles_pages.keys()) | set(chosen_layout.keys()) | set(page_scores.keys()))

    for page in pages_to_print:
        score = page_scores.get(page, None)
        score_txt = f" | Fitness: {score}" if score is not None else ""
        lay = chosen_layout.get(page, None)
        lay_txt = f" | Layout: {lay}" if lay is not None else " | Layout: ?"
        print(f"\nSeite {page}{lay_txt}{score_txt}")

        if page not in articles_pages or not articles_pages[page]:
            print("  (keine Box-Einträge)")
            continue

        for box in sorted(articles_pages[page]):
            items = articles_pages[page][box]

            # shell/hull anzeigen
            shell = chosen_shell.get((page, box), None)
            if isinstance(shell, list):
                shell_txt = f"{shell} (WARN: mehrere z=1)"
                shell_for_limits = shell[0]  # keep simple
            elif shell is None:
                shell_txt = "?"
                shell_for_limits = None
            else:
                shell_txt = str(shell)
                shell_for_limits = shell

            # hull limits holen
            hull_min = hull_max = None
            if isinstance(shell_for_limits, int) and shell_for_limits in hull_params:
                hull_min = hull_params[shell_for_limits]["min"]
                hull_max = hull_params[shell_for_limits]["max"]

            # Box-Füllgrad (Summe Längen)
            arts_in_box = [k for k in items.keys() if k != "<EMPTY>" and isinstance(k, int)]
            total_len = sum(article_length[a] for a in arts_in_box)

            # nur UNDER/OVER hinten dran, OK -> nix
            flag = _box_under_over(total_len, hull_min, hull_max)
            flag_txt = f" | {flag}" if flag else ""

            if "<EMPTY>" in items and len(items) == 1:
                print(f"  Box {box} | Shell: {shell_txt}: <EMPTY>{flag_txt}")
            else:
                def prio_key(k):
                    pr = items[k]
                    order = {"A": 0, "B": 1, "C": 2}
                    return (order.get(pr, 9), k if isinstance(k, int) else 10**9)

                arts = [k for k in items.keys() if k != "<EMPTY>"]
                arts = sorted(arts, key=prio_key)
                formatted = ", ".join([f"{a}({items[a]})" for a in arts])
                print(f"  Box {box} | Shell: {shell_txt}: {formatted}{flag_txt}")

    _h("RAW-DICTS (FALLS DU DEBUGGEN WILLST)", ch="-")
    print("articles_pages:")
    pprint({p: dict(b) for p, b in articles_pages.items()})

    print("\nchosen_layout:")
    pprint(dict(sorted(chosen_layout.items())))

    print("\nchosen_shell:")
    # nicer print
    pprint({f"{p},{b}": v for (p, b), v in sorted(chosen_shell.items())})

    print("\npage_scores:")
    pprint(dict(sorted(page_scores.items())))





