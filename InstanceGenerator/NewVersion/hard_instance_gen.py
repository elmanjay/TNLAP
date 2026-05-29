import json
import random
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Random-Generator
# ---------------------------------------------------------------------------
def _make_rngs(seed: Optional[int]):
    rng_main = random.Random(seed)
    return rng_main


# ---------------------------------------------------------------------------
# Instanz abspeichern
# ---------------------------------------------------------------------------
def save_instance_to_json(instance: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Canvas/Geometry
# ---------------------------------------------------------------------------
def _create_canvas(canvas_name: str = "A4", orientation: str = "portrait") -> Dict[str, float]:
    CANVAS = {
        "A4":               (210.0, 297.0),
        "A5":               (148.0, 210.0),
        "A3":               (297.0, 420.0),
        "NORDIC_BROADSHEET":(400.0, 570.0),
        "NORDIC_TABLOID":   (285.0, 400.0),
    }
    w, h = CANVAS[canvas_name]
    if orientation == "landscape":
        w, h = h, w
    return {"w": w, "h": h, "area": int(round(w * h, 0))}


Rect = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Layout-Hilfsfunktionen
# ---------------------------------------------------------------------------
def split_rect(rect: Rect, min_width_sr, min_height_sr, rng_sr) -> Tuple[Rect, Rect]:
    x, y, w, h = rect
    can_split_horizontally = w > 2 * min_width_sr
    can_split_vertically   = h > 2 * min_height_sr

    if can_split_horizontally and can_split_vertically:
        split_vertical = rng_sr.choice([True, False])
    elif can_split_horizontally:
        split_vertical = True
    elif can_split_vertically:
        split_vertical = False
    else:
        return rect, None

    if split_vertical:
        split = rng_sr.randint(min_width_sr, w - min_width_sr)
        r1 = (x, y, split, h)
        r2 = (x + split, y, w - split, h)
    else:
        split = rng_sr.randint(min_height_sr, h - min_height_sr)
        r1 = (x, y, w, split)
        r2 = (x, y + split, w, h - split)
    return r1, r2


def layout_to_tuple(layout: List[Rect]) -> Tuple[Rect, ...]:
    return tuple(sorted(layout))


def generate_layout(min_width_fc, min_height_fc, rng_gl) -> List[Rect]:
    cols = 6
    rows = 10
    # HARD: mehr Boxen pro Layout (3–5 statt 2–5)
    min_boxes = 3
    max_boxes = 5
    target_boxes = rng_gl.randint(min_boxes, max_boxes)
    rects    = [(0, 0, cols, rows)]
    attempts = 0
    max_attempts = 200
    while len(rects) < target_boxes and attempts < max_attempts:
        attempts += 1
        idx  = rng_gl.randrange(len(rects))
        rect = rects.pop(idx)
        r1, r2 = split_rect(rect, min_width_fc, min_height_fc, rng_gl)
        if r2 is None:
            rects.append(r1)
            continue
        rects.extend([r1, r2])
    return rects


def balanced_assignment(items, categories_fc, rng_ba):
    shuffled = items[:]
    rng_ba.shuffle(shuffled)
    return {
        item: categories_fc[i % len(categories_fc)]
        for i, item in enumerate(shuffled)
    }


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------
def create_instance(
        number_pages,
        number_article,
        p_type_fc,   # "A" -> sehr enge Hüllen  |  "B" -> etwas weniger eng
        seed,
):
    rng = _make_rngs(seed)
    pages  = list(range(1, number_pages + 1))
    canvas = _create_canvas("NORDIC_BROADSHEET", "portrait")

    # HARD: viele Layouts pro Seite (10–20)
    range_layouts_per_page = [10, 20]
    # HARD: viele Shells pro Box (6–15)
    shells_per_box = [6, 15]

    # ------------------------------------------------------------------
    # Layouts -> Pages  (Version 2: Überlappungen möglich)
    # ------------------------------------------------------------------
    layout_pool   = []
    pages_layouts = {}
    layout_counter = 1
    reuse_prob = 0.3
    for page in range(1, number_pages + 1):
        num_layouts = rng.randint(range_layouts_per_page[0], range_layouts_per_page[1])
        page_lay = set()
        while len(page_lay) < num_layouts:
            if layout_pool and rng.random() < reuse_prob:
                available = [lay for lay in layout_pool if lay not in page_lay]
                if available:
                    lay = rng.choice(available)
                    page_lay.add(lay)
                else:
                    lay = layout_counter
                    layout_counter += 1
                    layout_pool.append(lay)
                    page_lay.add(lay)
            else:
                lay = layout_counter
                layout_counter += 1
                layout_pool.append(lay)
                page_lay.add(lay)
        pages_layouts[page] = sorted(page_lay)

    layouts = sorted({lay for lays in pages_layouts.values() for lay in lays})

    # ------------------------------------------------------------------
    # Layouts erzeugen (eindeutig)
    # ------------------------------------------------------------------
    unique_layouts = set()
    all_layouts    = []
    while len(all_layouts) < len(layouts):
        min_width  = 2
        min_height = 3
        layout     = generate_layout(min_width, min_height, rng)
        layout_key = layout_to_tuple(layout)
        if layout_key not in unique_layouts:
            unique_layouts.add(layout_key)
            all_layouts.append(layout)

    geometry_layout_box = {
        i + 1: {
            j + 1: {"x": x, "y": y, "w": w, "h": h, "character": w * h * 40 * 10}
            for j, (x, y, w, h) in enumerate(lay)
        }
        for i, lay in enumerate(all_layouts)
    }
    box_layouts = {
        i + 1: list(range(1, len(lay) + 1))
        for i, lay in enumerate(all_layouts)
    }

    # ------------------------------------------------------------------
    # Shells -> Boxen  +  Shell-Parameter (Typ A/B bleibt erhalten,
    #                                       aber beide jetzt auf hard-Niveau)
    # ------------------------------------------------------------------
    counter      = 1
    shells_layout_box = {}
    shell_params     = {}
    number_shells    = []

    for layout_id, boxes in geometry_layout_box.items():
        shells_layout_box[layout_id] = {}
        for box_id, box_data in boxes.items():
            num_shells = rng.randint(shells_per_box[0], shells_per_box[1])
            character  = box_data["character"]
            shells_lst = list(range(counter, counter + num_shells))
            shells_layout_box[layout_id][box_id] = shells_lst

            for shell_id in shells_lst:
                # HARD: max_val aus engerem oberen Bereich gezogen (0.75–1.0 statt 0.5–1.0)
                max_val = round(character * rng.uniform(0.75, 1.0))

                # Typ A/B: bleibt semantisch erhalten, beide auf hard-Niveau hochgezogen
                if p_type_fc == "B":
                    lo, hi = 0.92, 0.96   # war 0.75–0.85 → jetzt eng
                else:                      # Typ A
                    lo, hi = 0.96, 0.99   # war 0.90–0.95 → jetzt sehr eng
                fac     = rng.uniform(lo, hi)
                min_val = round(max_val * fac)

                shell_params[shell_id] = {"min": min_val, "max": max_val}
                number_shells.append(shell_id)
            counter += num_shells

    # ------------------------------------------------------------------
    # Artikel -> Shells  (Methode 2: Kategorien, keine ungenutzten Shells)
    # HARD: mehr Kategorien -> mehr Shells je Artikel
    # ------------------------------------------------------------------
    article = list(range(1, number_article + 1))
    shells  = list(range(1, len(shell_params) + 1))

    num_categories      = min(len(article), 20)
    categories          = list(range(num_categories))
    article_to_category = balanced_assignment(article, categories, rng)
    shell_to_category   = balanced_assignment(shells,  categories, rng)

    category_to_shells = {}
    for s, cat in shell_to_category.items():
        category_to_shells.setdefault(cat, []).append(s)

    shells_article = {
        art: category_to_shells.get(cat, [])
        for art, cat in article_to_category.items()
    }
    for a in shells_article:
        shells_article[a] = sorted(shells_article[a])
    shells_article = dict(sorted(shells_article.items()))

    # ------------------------------------------------------------------
    # Artikellängen
    # ------------------------------------------------------------------
    article_lengths = {a: rng.randint(500, 20000) for a in article}

    # ------------------------------------------------------------------
    # Artikelpriorität
    # ------------------------------------------------------------------
    n      = len(article)
    a_prio = rng.uniform(0.10, 0.2)
    b_prio = rng.uniform(0.3,  0.4)
    a_count = max(1, int(round(a_prio * n)))
    b_count = max(1, int(round(b_prio * n)))
    remaining = article.copy()
    A = set(rng.sample(remaining, a_count))
    for a in A:
        remaining.remove(a)
    B = set(rng.sample(remaining, b_count))
    for b in B:
        remaining.remove(b)
    C = set(remaining)
    article_priority = {
        a: ("A" if a in A else "B" if a in B else "C")
        for a in article
    }

    # ------------------------------------------------------------------
    # section (1-zu-1 wie bisher)
    # ------------------------------------------------------------------
    sections        = [1]
    article_sections = {1: article.copy()}
    sections_page    = {p: sections.copy() for p in pages}

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    return {
        "pages":                pages,
        "layouts":              layouts,
        "article":              article,
        "shells":                shells,
        "layouts_pages":        pages_layouts,
        "box_layouts":          box_layouts,
        "geometry_layout_box":  geometry_layout_box,
        "shells_layout_box":      shells_layout_box,
        "shells_article":         shells_article,
        "article_length":       article_lengths,
        "shell_params":          shell_params,
        "article_priority":     article_priority,
        "sections":              sections,
        "article_sections":      article_sections,
        "sections_page":          sections_page,
    }


# ---------------------------------------------------------------------------
# Aufruf
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pages_gl   = 10
    article_gl = 70
    seed_gl    = 232
    for p_type in ["A", "B"]:
        instance = create_instance(
            number_pages=pages_gl,
            number_article=article_gl,
            p_type_fc=p_type,
            seed=seed_gl,
        )
        name = f"HardP{pages_gl}A{article_gl}V1({p_type})({seed_gl}).json"
        save_instance_to_json(instance, name)
        print(f"✅  {name} gespeichert.")