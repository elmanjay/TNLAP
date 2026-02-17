import json
import random
import math
from typing import Dict, List, Tuple

def create_instance(
    p_type: str,
    number_pages: int,
    number_layouts: int,
    number_article: int,
    number_hulls: int = None,           # wird ignoriert, wenn use_explicit_hulls=False
    min_boxes: int = 2,
    max_boxes: int = 5,
    min_layouts: int = 1,
    max_layouts: int = 3,
    max_hulls: int = 15,                # Max. Shells je Box/Artikel
    use_explicit_hulls: bool = False,   # True -> benutze number_hulls; False -> n_s = 20 * n_a
    seed: int = None
):
    """
    Erzeugt eine Instanz gemäß Spezifikation:
    - n_s = 20 * n_a (falls use_explicit_hulls=False), sonst number_hulls
    - Boxen pro Layout: [min_boxes, max_boxes]
    - Shells je (Layout, Box): k in [1, ceil(n_s/10)], gedeckelt durch max_hulls
    - Shells je Artikel:       k in [1, ceil(n_s/15)], gedeckelt durch max_hulls
    - Shellkapazitäten: max in [500, 12000], min in [0.8*max, 0.9*max]
    - Artikellängen: 1200–12000
    - Prioritäten: 'A', 'B', 'C'
    """

    if seed is not None:
        random.seed(seed)
        rng_min  = random.Random(seed + 99999)

    # --- Validierungen ---
    if number_pages <= 0 or number_layouts <= 0 or number_article <= 0:
        raise ValueError("number_pages, number_layouts und number_article müssen > 0 sein.")
    if not (1 <= min_boxes <= max_boxes):
        raise ValueError("Erwarte 1 <= min_boxes <= max_boxes.")
    if not (1 <= min_layouts <= max_layouts <= number_layouts):
        raise ValueError("Erwarte 1 <= min_layouts <= max_layouts <= number_layouts.")
    if max_hulls <= 0:
        raise ValueError("max_hulls muss > 0 sein.")

    # --- Grundmengen ---
    pages   = list(range(1, number_pages + 1))
    layouts = list(range(1, number_layouts + 1))
    article = list(range(1, number_article + 1))

    # --- n_s bestimmen (Shells/Hüllen) ---
    if use_explicit_hulls:
        if number_hulls is None or number_hulls <= 0:
            raise ValueError("Wenn use_explicit_hulls=True, muss number_hulls > 0 sein.")
        n_s = number_hulls
    else:
        n_s = int(0.8 * number_article)  # Article-to-shell ratio

    hulls = list(range(1, n_s + 1))

    # --- Step 1: Layouts pro Seite ---
    layouts_pages: Dict[int, List[int]] = {}
    for p in pages:
        num_l = random.randint(min_layouts, max_layouts)
        layouts_pages[p] = sorted(random.sample(layouts, num_l))

    # --- Step 2: Boxen pro Layout ---
    box_layouts: Dict[int, List[int]] = {}
    for m in layouts:
        b = random.randint(min_boxes, max_boxes)
        box_layouts[m] = list(range(1, b + 1))

    # --- Step 5: Shell-Parameter (min/max) ---
    hull_params: Dict[int, Dict[str, int]] = {}
    for s in hulls:
        maximum = random.randint(500, 12000)

        if p_type == "B":
            minimum = rng_min.randint(int(maximum * 0.75), int(maximum * 0.85))
        else:
            minimum = rng_min.randint(int(maximum * 0.90), int(maximum * 0.95))
        hull_params[s] = {"min": minimum, "max": maximum}

    # --- Step 3: Shell-Zuweisungen zu (Layout, Box) ---
    per_box_upper = min(max_hulls, max(1, math.ceil(n_s / 10)))
    hull_layout_box: Dict[int, Dict[int, List[int]]] = {}
    for m in layouts:
        hull_layout_box[m] = {}
        for b in box_layouts[m]:
            k_s = random.randint(1, per_box_upper)
            k_s = min(k_s, len(hulls))
            selected = sorted(random.sample(hulls, k_s))
            hull_layout_box[m][b] = selected

    # --- Step 4: Shell-Zuweisungen zu Artikeln ---
    per_article_upper = min(max_hulls, max(1, math.ceil(n_s / 15)))
    hull_article: Dict[int, List[int]] = {}
    for a in article:
        k_s = random.randint(1, per_article_upper)
        k_s = min(k_s, len(hulls))
        hull_article[a] = sorted(random.sample(hulls, k_s))

    # --- Step 6: Artikel-Parameter (Länge & Priorität) ---
    article_length: Dict[int, int] = {a: random.randint(1200, 12000) for a in article}

    # Priorität: A, B, C
    # z. B. 15% A, 10% B, Rest C (du kannst das leicht ändern)
    n = len(article)
    n_A = max(1, int(0.05 * n))
    n_B = max(1, int(0.10 * n))
    A_set = set(random.sample(article, n_A))
    remaining = [a for a in article if a not in A_set]
    B_set = set(random.sample(remaining, n_B))
    article_priority: Dict[int, str] = {}
    for a in article:
        if a in A_set:
            article_priority[a] = 'A'
        elif a in B_set:
            article_priority[a] = 'B'
        else:
            article_priority[a] = 'C'

    # --- Resort-Logik ---
    resorts = [1]
    article_resorts = {1: article.copy()}
    resort_page = {p: resorts.copy() for p in pages}

    return {
        "pages": pages,
        "layouts": layouts,
        "article": article,
        "hulls" : hulls,
        "layouts_pages": layouts_pages,
        "box_layouts": box_layouts,
        "hull_layout_box": hull_layout_box,
        "hull_article": hull_article,
        "article_length": article_length,
        "hull_params": hull_params,
        "article_priority": article_priority,
        "resorts": resorts,
        "article_resorts": article_resorts,
        "resort_page": resort_page
    }

def save_instance_to_json(instance, filename):
    with open(filename, 'w') as json_file:
        json.dump(instance, json_file, indent=4)

def parse_json_from_file(file_name):
    with open(file_name, "r") as f:
        json_data = json.load(f)

    pages = json_data["pages"]
    article = json_data["article"]
    layouts = json_data["layouts"]
    resorts = json_data["resorts"]
    article_resorts = {int(k): v for k, v in json_data["article_resorts"].items()}
    resort_page = {int(k): v for k, v in json_data["resort_page"].items()}
    layouts_pages = {int(k): v for k, v in json_data["layouts_pages"].items()}
    box_layouts = {int(k): v for k, v in json_data["box_layouts"].items()}
    hull_layout_box = {
        int(layout): {int(box): hulls for box, hulls in boxes.items()}
        for layout, boxes in json_data["hull_layout_box"].items()
    }
    hull_article = {int(k): v for k, v in json_data["hull_article"].items()}
    article_length = {int(k): v for k, v in json_data["article_length"].items()}
    article_priority = {int(k): v for k, v in json_data["article_priority"].items()}
    hull_params = {int(k): v for k, v in json_data["hull_params"].items()}

    return (pages, article, layouts, resorts, article_resorts, resort_page,
            layouts_pages, box_layouts, hull_layout_box, hull_article,
            article_length, hull_params, article_priority)

if __name__ == "__main__":

    pages_list = [5,10,20,30]
    numer_article = [2,3,5,7,10]
    number_layouts= [5,10]
    max_layouts_list = [5,10,15]
    max_hulls_list = [10,20,30]


    # for i in range(1, 2):
    #     for pages in pages_list:
    #         for article in [j*pages for j in numer_article ]:
    #             for layout_param in [l*pages for l in number_layouts]:
    #                 for max_layouts_param in max_layouts_list:
    #                     for max_hulls_param in max_hulls_list:
    #                         print(layout_param)
    #                         print(max_layouts_param)
    #                         number_pages = pages
    #                         number_article = article
    #                         instance = create_instance(
    #                             number_pages=number_pages,
    #                             number_layouts=layout_param,
    #                             number_article=number_article,
    #                             min_boxes=2,
    #                             max_boxes=5,
    #                             min_layouts=1,
    #                             max_layouts=max_layouts_param,
    #                             max_hulls=max_hulls_param,
    #                             use_explicit_hulls=False,
    #                             seed=42 + i
    #                         )

    prefix = "L"
    number_pages = 30
    number_article = 180
    problem_type = "A"
    for i in range(1,2):
        instance = create_instance(
            p_type = problem_type,
            number_pages=number_pages,
            number_layouts=100,
            number_article=number_article,
            min_boxes=2,
            max_boxes=5,
            min_layouts=5,
            max_layouts=10,
            max_hulls=int(0.8 * number_article),
            use_explicit_hulls=False,
            seed=42 + i
        )
        name = f"{prefix}{number_pages}P{number_article}A{i}({problem_type}).json"
        save_instance_to_json(instance, name)
