import json
import math
import random
from typing import Dict, List, Optional, Tuple


#Random-Generator:
def _make_rngs(seed: Optional[int]): 
    rng_main = random.Random(seed)
    rng_min = random.Random(seed + 99999) if seed is not None else random.Random()
    return rng_main, rng_min

#Instanz abspeichern:
def save_instance_to_json(instance: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=4, ensure_ascii=False)

#Rechteck: (x, y, width, height) für Layout
Rect = Tuple[int, int, int, int]

#Hilfsfunktion Erstellung Layout:
def split_rect(rect: Rect, min_width, min_height, rng) -> Tuple[Rect, Rect]:
    x, y, w, h = rect

    # Prüfe, ob ein Split überhaupt möglich ist
    can_split_horizontally = w > 2 * min_width
    can_split_vertically = h > 2 * min_height

    if can_split_horizontally and can_split_vertically:
        split_vertical = rng.choice([True, False])
    elif can_split_horizontally:
        split_vertical = True
    elif can_split_vertically:
        split_vertical = False
    else:
        # Nicht teilbar
        return rect, None

    if split_vertical:
        # Wähle Split-Punkt, sodass beide Seiten >= min_width
        split = rng.randint(min_width, w - min_width)
        r1 = (x, y, split, h)
        r2 = (x + split, y, w - split, h)
    else:
        # Split horizontal, beide Seiten >= min_height
        split = rng.randint(min_height, h - min_height)
        r1 = (x, y, w, split)
        r2 = (x, y + split, w, h - split)

    return r1, r2

#Hilfsfunktion zum Prüfen, ob Layouts doppelt generiert: 
def layout_to_tuple(layout: List[Rect]) -> Tuple[Rect, ...]:
    return tuple(sorted(layout))

#Funktion zur Erstellung des Layouts:
def generate_layout(min_width_fc, min_height_fc, rng) -> List[Rect]:
    cols = 6
    rows = 10 #hier dann ggf. 100
    min_boxes = 5
    max_boxes = 5
    target_boxes = rng.randint(min_boxes, max_boxes)
    rects = [(0, 0, cols, rows)]
    attempts = 0
    max_attempts = 50 #kann sonst in eine Endlosschleife geraten, ob 50 oder weniger müsste man noch testen aber wäre nur für LZ relevant

    while len(rects) < target_boxes and attempts < max_attempts:
        attempts += 1
        idx = rng.randrange(len(rects))
        rect = rects.pop(idx)
        
        r1, r2 = split_rect(rect, min_width_fc, min_height_fc, rng)
        if r2 is None:
            # Rechteck nicht teilbar, zurücklegen
            rects.append(r1)
            continue
        rects.extend([r1, r2])

    return rects

#Erstelle Instanz:
def create_instance(
        number_pages,
        number_layouts,
        range_layouts_per_page,
        number_article,
        seed,
):
    
    rng, rng_min = _make_rngs(seed) #random seed

    pages = list(range(1, number_pages + 1)) #Seiten als Liste
    layouts = list(range(1, number_layouts + 1)) #Layouts als Liste

    #Zuweisung Layouts -> Pages
    pages_layouts = {p: sorted(rng.sample(layouts, rng.randint(range_layouts_per_page[0], range_layouts_per_page[1]))) for p in pages}
    "{page: [verf. Layouts]}"

    #Erstellung der Layouts:
    unique_layouts = set()
    all_layouts = []
    while len(all_layouts) < number_layouts:
        min_width = 2
        min_height = 3 #falls mit 100 Zeilen -> height = 30
        layout = generate_layout(min_width, min_height, rng) 
        layout_key = layout_to_tuple(layout)
        if layout_key not in unique_layouts:
            unique_layouts.add(layout_key)
            all_layouts.append(layout)
    #ins Format bringen:        
    geometry_layout_box = {
        i + 1: {
            j + 1: {"x": x, "y": y, "w": w, "h": h, "character": w * h * 10 * 40} #stimmt das? Character soll Anz. Zeichen in der Box beschreiben
            for j, (x, y, w, h) in enumerate(lay) #*10 nur, falls wir mit 10 Zeilen und nicht 100 rechnen!
        }
        for i, lay in enumerate(all_layouts)
    }
    "{Layout 1: {Box 1: {x,y,w,h,character}, Box 2: {x,y,w,h,character}}, Layout 2: {...}, ...}"

    #Zuweisung Shells zu Boxen:

    #nächster Schritt




    #Zuweisung Artikellänge
    article = list(range(1, number_article + 1)) #Artikel als Liste
    article_lengths = {a: rng.randint(400, 1300) for a in article} #1300 als Max, 400 als Min, weil wir sagen minimale Box hat 2 Spalten und 33 Zeilen
    "{article: [Länge d. Artikels]}"
    
    #Zuweisung Artikelpriorität: Hier als prozentualer Bereich -> testen was das Modell "schwerer" macht
    n = len(article)
    a_prio = rng.uniform(0.10, 0.2) #10-20%
    b_prio = rng.uniform(0.3, 0.4) #30-40%
    #c_prio = 1 - (a_prio + b_prio) #Rest
    a_count = max(1, int(round(a_prio * n))) #min. 1 Artikel je Klasse
    b_count = max(1, int(round(b_prio * n)))
    #c_count = max(1, n - (a_count + b_count))
    remaining = article.copy()
    A = set(rng.sample(remaining, a_count))
    for a in A: remaining.remove(a)
    B = set(rng.sample(remaining, b_count))
    for b in B: remaining.remove(b)
    C = set(remaining)
    article_priority = {a: ("A" if a in A else "B" if a in B else "C") for a in article}




#kleine Instanz zum "ausprobieren":
pages_gl = 5
layouts_gl = 10
layouts_per_page_range = [2, 5] #[min,max]
article_gl = 20
seed_gl = 42
create_instance(pages_gl, layouts_gl, layouts_per_page_range, article_gl, seed_gl)