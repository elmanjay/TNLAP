import json
import math
import random
from typing import Dict, List, Optional


# ----------------------------
# IO
# ----------------------------
def save_instance_to_json(instance: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=4, ensure_ascii=False)


def parse_json_from_file(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    canvas = json_data["canvas"]
    pages = json_data["pages"]
    article = json_data["article"]
    layouts = json_data["layouts"]
    resorts = json_data["resorts"]

    article_resorts = {int(k): v for k, v in json_data["article_resorts"].items()}
    resort_page = {int(k): v for k, v in json_data["resort_page"].items()}
    layouts_pages = {int(k): v for k, v in json_data["layouts_pages"].items()}
    box_layouts = {int(k): v for k, v in json_data["box_layouts"].items()}

    geometry_layout_box = {
        int(l): {int(b): g for b, g in boxes.items()}
        for l, boxes in json_data["geometry_layout_box"].items()
    }

    hull_layout_box = {
        int(l): {int(b): v for b, v in boxes.items()}
        for l, boxes in json_data["hull_layout_box"].items()
    }

    hull_article = {int(k): v for k, v in json_data["hull_article"].items()}
    article_length = {int(k): v for k, v in json_data["article_length"].items()}
    article_priority = {int(k): v for k, v in json_data["article_priority"].items()}
    hull_params = {int(k): v for k, v in json_data["hull_params"].items()}

    return (
        canvas,
        pages,
        article,
        layouts,
        resorts,
        article_resorts,
        resort_page,
        layouts_pages,
        box_layouts,
        geometry_layout_box,
        hull_layout_box,
        hull_article,
        article_length,
        hull_params,
        article_priority,
    )


# ----------------------------
# Helper / Validation
# ----------------------------
def _validate_inputs(
    number_pages: int,
    number_layouts: int,
    number_article: int,
    number_hulls_per_layout: int,
    min_boxes: int,
    max_boxes: int,
    min_layouts: int,
    max_layouts: int,
    max_hulls: int,
) -> None:
    if number_pages <= 0 or number_layouts <= 0 or number_article <= 0:
        raise ValueError("number_pages, number_layouts und number_article müssen > 0 sein.")
    if number_hulls_per_layout <= 0:
        raise ValueError("number_hulls_per_layout muss > 0 sein.")
    if not (1 <= min_boxes <= max_boxes <= 12):
        raise ValueError("Erwarte 1 <= min_boxes <= max_boxes <= 12.")
    if not (1 <= min_layouts <= max_layouts <= number_layouts):
        raise ValueError("Erwarte 1 <= min_layouts <= max_layouts <= number_layouts.")
    if max_hulls <= 0:
        raise ValueError("max_hulls muss > 0 sein.")


def _make_rngs(seed: Optional[int]):
    rng_main = random.Random(seed)
    rng_min = random.Random(seed + 99999) if seed is not None else random.Random()
    return rng_main, rng_min


# ----------------------------
# Canvas / Geometry
# ----------------------------
def _create_canvas(canvas_name: str = "A4", orientation: str = "portrait") -> Dict[str, float]:
    """
    Nur Seitenformat (ohne margins).
    """
    CANVAS = {
        "A4": (210.0, 297.0),
        "A5": (148.0, 210.0),
        "A3": (297.0, 420.0),

        # nordisches Zeitungsformat (Broadsheet)
        "NORDIC_BROADSHEET": (400.0, 570.0),
        "NORDIC_TABLOID": (285.0, 400.0),
    }

    w, h = CANVAS[canvas_name]
    if orientation == "landscape":
        w, h = h, w

    return {"w": w, "h": h, "area": int(round(w * h, 0))}


def _random_spans_sum_to(cols: int, parts: int, rng: random.Random) -> List[int]:
    """
    Liefert 'parts' viele positive Integer-Spans, deren Summe = cols ist.
    Beispiel: cols=6, parts=3 -> [2,1,3]
    """
    if parts <= 1:
        return [cols]
    cuts = sorted(rng.sample(range(1, cols), parts - 1))
    spans = []
    prev = 0
    for c in cuts + [cols]:
        spans.append(c - prev)
        prev = c
    return spans


def _create_geometry_layout_box(
    layouts: List[int],
    box_layouts: Dict[int, List[int]],
    canvas: Dict[str, float],
    rng: random.Random,
    grid_cols: int = 6,       # Gesamt-Spalten im Raster (pro Zeile)
    gutter: float = 4.0,      # Steg zwischen Spalten (mm)
    p_two_rows: float = 0.5,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    6-Spalten-Zeitungsraster ohne margins.

    Pro Box speichern wir:
      - row: Zeilenindex
      - columns: WIE VIELE Spalten breit die Box ist (Span: 1..grid_cols)
    """
    cw, ch = float(canvas["w"]), float(canvas["h"])

    if grid_cols <= 0:
        raise ValueError("grid_cols muss > 0 sein.")
    if cw <= (grid_cols - 1) * gutter:
        gutter = max(0.0, (cw / grid_cols) * 0.15)

    col_w = (cw - (grid_cols - 1) * gutter) / grid_cols

    out: Dict[int, Dict[int, Dict[str, float]]] = {}

    for l in layouts:
        boxes = box_layouts[l]
        n = len(boxes)

        want_two_rows = (rng.random() < p_two_rows)

        # 1 oder 2 Zeilen; jede Zeile belegt exakt grid_cols Spalten
        if n <= 1:
            row_counts = [n]
        else:
            if want_two_rows:
                possible_splits = [
                    [top, n - top]
                    for top in range(1, min(grid_cols, n))
                    if 1 <= (n - top) <= grid_cols
                ]
                row_counts = (
                    rng.choice(possible_splits)
                    if possible_splits
                    else [min(grid_cols, n), max(0, n - grid_cols)]
                )
                row_counts = [c for c in row_counts if c > 0]
            else:
                row_counts = [n] if n <= grid_cols else [grid_cols, n - grid_cols]

        row_h = ch / len(row_counts)

        out[l] = {}
        idx = 0

        for r, cnt in enumerate(row_counts):
            spans = _random_spans_sum_to(grid_cols, cnt, rng)

            y = r * row_h
            col_cursor = 0

            for span in spans:
                b = boxes[idx]

                x = col_cursor * (col_w + gutter)
                w = span * col_w + (span - 1) * gutter
                h = row_h

                out[l][b] = {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": int(round(w * h, 0)),

                    # genau wie gewünscht:
                    "row": r,
                    "columns": span,  # <- Box-Spaltenbreite (nicht immer 6)
                }

                col_cursor += span
                idx += 1

    return out


# ----------------------------
# Generation steps
# ----------------------------
def _assign_layouts_to_pages(
    pages: List[int],
    layouts: List[int],
    min_layouts: int,
    max_layouts: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    return {
        p: sorted(rng.sample(layouts, rng.randint(min_layouts, max_layouts)))
        for p in pages
    }


def _create_boxes_per_layout(
    layouts: List[int],
    min_boxes: int,
    max_boxes: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    return {l: list(range(1, rng.randint(min_boxes, max_boxes) + 1)) for l in layouts}


# ----------------------------
# Hulls per Layout (flattened)
# ----------------------------
def _global_hull_id(layout_id: int, local_h: int, number_hulls_per_layout: int) -> int:
    return (layout_id - 1) * number_hulls_per_layout + local_h


def _create_hull_params_per_layout_flat(
    layouts: List[int],
    number_hulls_per_layout: int,
    p_type: str,
    rng_main: random.Random,
    rng_min: random.Random,
    layout_jitter: float = 0.12,
) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}

    lo = 0.75 if p_type == "B" else 0.90
    hi = 0.85 if p_type == "B" else 0.95

    base_min, base_max = 500, 12000

    for l in layouts:
        factor = 1.0 + rng_main.uniform(-layout_jitter, layout_jitter)
        mn_range = int(base_min * factor)
        mx_range = int(base_max * factor)

        H = number_hulls_per_layout
        for idx in range(H):
            local_h = idx + 1
            t = 0.0 if H == 1 else idx / (H - 1)

            mx = int(mn_range + t * (mx_range - mn_range))
            mx = int(mx * rng_main.uniform(0.92, 1.08))
            mn = rng_min.randint(int(mx * lo), int(mx * hi))

            gh = _global_hull_id(l, local_h, number_hulls_per_layout)
            out[gh] = {"min": mn, "max": mx}

    return out


def _assign_similar_hulls_to_layout_boxes_flat(
    layouts: List[int],
    box_layouts: Dict[int, List[int]],
    number_hulls_per_layout: int,
    max_hulls: int,
    rng: random.Random,
    cluster_width: int = 6,
    p_reuse_center: float = 0.7,
) -> Dict[int, Dict[int, List[int]]]:
    out: Dict[int, Dict[int, List[int]]] = {}

    H = number_hulls_per_layout
    if H <= 0:
        return {l: {b: [] for b in box_layouts[l]} for l in layouts}

    per_box_upper = min(max_hulls, max(math.ceil(H / 15), math.ceil(H / 10), 1))
    kmax = min(per_box_upper, H)

    for l in layouts:
        favored_centers = [rng.randint(1, H) for _ in range(max(1, min(5, H)))]

        out[l] = {}
        for b in box_layouts[l]:
            k = rng.randint(1, kmax)
            c = rng.choice(favored_centers) if (rng.random() < p_reuse_center) else rng.randint(1, H)

            half = max(1, cluster_width // 2)
            left = max(1, c - half)
            right = min(H, c + half)

            segment = list(range(left, right + 1))

            while len(segment) < k and (left > 1 or right < H):
                if left > 1:
                    left -= 1
                    segment.insert(0, left)
                if len(segment) >= k:
                    break
                if right < H:
                    right += 1
                    segment.append(right)

            chosen_local = rng.sample(segment, min(k, len(segment)))
            chosen_global = sorted(_global_hull_id(l, lh, H) for lh in chosen_local)
            out[l][b] = chosen_global

    return out


# ----------------------------
# Articles
# ----------------------------
def _create_article_lengths(article: List[int], rng: random.Random) -> Dict[int, int]:
    return {a: rng.randint(1200, 12000) for a in article}


def _assign_hulls_to_articles(
    article: List[int],
    hulls: List[int],
    rng: random.Random,
    min_hulls_per_article: int = 10,
    max_hulls_per_article: int = 20,
) -> Dict[int, List[int]]:
    if min_hulls_per_article <= 0 or max_hulls_per_article <= 0:
        raise ValueError("min/max_hulls_per_article müssen > 0 sein.")
    if min_hulls_per_article > max_hulls_per_article:
        raise ValueError("min_hulls_per_article darf nicht > max_hulls_per_article sein.")

    out: Dict[int, List[int]] = {}

    if not hulls:
        return {a: [] for a in article}

    for a in article:
        k = rng.randint(min_hulls_per_article, max_hulls_per_article)
        k = min(k, len(hulls))
        out[a] = sorted(rng.sample(hulls, k))

    return out


def _assign_article_priorities(article: List[int], rng: random.Random) -> Dict[int, str]:
    n = len(article)
    A = set(rng.sample(article, max(1, int(0.10 * n))))
    B = set(rng.sample([a for a in article if a not in A], max(1, int(0.40 * n))))
    return {a: ("A" if a in A else "B" if a in B else "C") for a in article}


# ----------------------------
# Main
# ----------------------------
def create_instance(
    p_type: str,
    number_pages: int,
    number_layouts: int,
    number_article: int,
    number_hulls_per_layout: int,
    min_boxes: int = 2,
    max_boxes: int = 5,
    min_layouts: int = 1,
    max_layouts: int = 3,
    max_hulls: int = 15,
    seed: int = None,
) -> dict:
    _validate_inputs(
        number_pages, number_layouts, number_article,
        number_hulls_per_layout,
        min_boxes, max_boxes,
        min_layouts, max_layouts,
        max_hulls,
    )

    rng, rng_min = _make_rngs(seed)

    pages = list(range(1, number_pages + 1))
    layouts = list(range(1, number_layouts + 1))
    article = list(range(1, number_article + 1))

    canvas = _create_canvas("NORDIC_BROADSHEET", "portrait")

    layouts_pages = _assign_layouts_to_pages(pages, layouts, min_layouts, max_layouts, rng)
    box_layouts = _create_boxes_per_layout(layouts, min_boxes, max_boxes, rng)

    geometry_layout_box = _create_geometry_layout_box(
        layouts, box_layouts, canvas, rng,
        grid_cols=6,
        gutter=4.0,
        p_two_rows=0.5,
    )

    total_hulls = number_layouts * number_hulls_per_layout
    hulls = list(range(1, total_hulls + 1))

    hull_params = _create_hull_params_per_layout_flat(
        layouts=layouts,
        number_hulls_per_layout=number_hulls_per_layout,
        p_type=p_type,
        rng_main=rng,
        rng_min=rng_min,
    )

    hull_layout_box = _assign_similar_hulls_to_layout_boxes_flat(
        layouts=layouts,
        box_layouts=box_layouts,
        number_hulls_per_layout=number_hulls_per_layout,
        max_hulls=max_hulls,
        rng=rng,
        cluster_width=6,
        p_reuse_center=0.7,
    )

    article_length = _create_article_lengths(article, rng)
    hull_article = _assign_hulls_to_articles(article, hulls, rng, min_hulls_per_article=10, max_hulls_per_article=20)
    article_priority = _assign_article_priorities(article, rng)

    resorts = [1]
    article_resorts = {1: article.copy()}
    resort_page = {p: resorts.copy() for p in pages}

    return {
        "canvas": canvas,
        "pages": pages,
        "layouts": layouts,
        "article": article,
        "hulls": hulls,
        "layouts_pages": layouts_pages,
        "box_layouts": box_layouts,
        "geometry_layout_box": geometry_layout_box,
        "hull_layout_box": hull_layout_box,
        "hull_article": hull_article,
        "article_length": article_length,
        "hull_params": hull_params,
        "article_priority": article_priority,
        "resorts": resorts,
        "article_resorts": article_resorts,
        "resort_page": resort_page,
    }


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    prefix = "G"
    number_pages = 30
    number_article = 180
    problem_type = "A"

    for i in range(1, 2):
        instance = create_instance(
            p_type=problem_type,
            number_pages=number_pages,
            number_layouts=100,
            number_article=number_article,
            number_hulls_per_layout=30,
            min_boxes=2,
            max_boxes=5,
            min_layouts=7,
            max_layouts=15,
            max_hulls=1000,
            seed=42 + i,
        )

        name = f"{prefix}{number_pages}P{number_article}A{i}({problem_type}).json"
        save_instance_to_json(instance, name)



