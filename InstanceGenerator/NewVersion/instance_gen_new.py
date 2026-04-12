import json
import random
from typing import Dict, List, Optional, Tuple


#Random-Generator:
def _make_rngs(seed: Optional[int]): 
    rng_main = random.Random(seed)
    #rng_min = random.Random(seed + 99999) if seed is not None else random.Random()
    return rng_main #,rng_min

#Instanz abspeichern (wie bei dir):
def save_instance_to_json(instance: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=4, ensure_ascii=False)

#Canvas/Geometry (wie bei dir):
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

#Rechteck: (x, y, width, height) für Layout
Rect = Tuple[int, int, int, int]

#Hilfsfunktion Erstellung Layout:
def split_rect(rect: Rect, min_width_sr, min_height_sr, rng_sr) -> Tuple[Rect, Rect]:
    x, y, w, h = rect
    # Prüfe, ob ein Split überhaupt möglich ist
    can_split_horizontally = w > 2 * min_width_sr
    can_split_vertically = h > 2 * min_height_sr

    if can_split_horizontally and can_split_vertically:
        split_vertical = rng_sr.choice([True, False])
    elif can_split_horizontally:
        split_vertical = True
    elif can_split_vertically:
        split_vertical = False
    else:
        # Nicht teilbar
        return rect, None
    if split_vertical:
        # Wähle Split-Punkt, sodass beide Seiten >= min_width
        split = rng_sr.randint(min_width_sr, w - min_width_sr)
        r1 = (x, y, split, h)
        r2 = (x + split, y, w - split, h)
    else:
        # Split horizontal, beide Seiten >= min_height
        split = rng_sr.randint(min_height_sr, h - min_height_sr)
        r1 = (x, y, w, split)
        r2 = (x, y + split, w, h - split)
    return r1, r2

#Hilfsfunktion zum Prüfen, ob Layouts doppelt generiert: 
def layout_to_tuple(layout: List[Rect]) -> Tuple[Rect, ...]:
    return tuple(sorted(layout))

#Funktion zur Erstellung des Layouts:
def generate_layout(min_width_fc, min_height_fc, rng_gl) -> List[Rect]:
    cols = 6
    rows = 10 #hier dann ggf. 100 #Anpassung bei 100*6 Layout
    min_boxes = 2
    max_boxes = 5
    target_boxes = rng_gl.randint(min_boxes, max_boxes)
    rects = [(0, 0, cols, rows)]
    attempts = 0
    max_attempts = 200 #kann sonst in eine Endlosschleife geraten, ob weniger müsste man noch testen aber wäre nur für LZ relevant
    while len(rects) < target_boxes and attempts < max_attempts:
        attempts += 1
        idx = rng_gl.randrange(len(rects))
        rect = rects.pop(idx)   
        r1, r2 = split_rect(rect, min_width_fc, min_height_fc, rng_gl)
        if r2 is None:
            # Rechteck nicht teilbar, zurücklegen
            rects.append(r1)
            continue
        rects.extend([r1, r2])
    return rects

#Hilfsfunktion Zuweisung Artikel -> Shells:
def balanced_assignment(items, categories_fc, rng_ba): #gleich große Kategorien und keine leer
    shuffled = items[:]
    rng_ba.shuffle(shuffled)
    return {
        item: categories_fc[i % len(categories_fc)]
        for i, item in enumerate(shuffled)
    }

#Erstelle Instanz: Hauptfunktion
def create_instance(
        number_pages,
        range_layouts_per_page,
        shells_per_box_fc,
        number_article,
        shells_per_article_fc, #wird nur für bestimme Methode benötigt
        p_type_fc,
        seed,
):  
    rng = _make_rngs(seed) #random seed  #,rng_min 
    pages = list(range(1, number_pages + 1)) #Seiten als Liste
    canvas = _create_canvas("NORDIC_BROADSHEET", "portrait") #wie bei dir

    #Zuweisung Layouts -> Pages: Zwei unterschiedliche Methoden, einfach """...""" raus/rein nehmen
    """
    #Version 1: Je Seite exklusive Layouts, Achtung: Keine gleichen Layouts jemals, sonst Fkt. umstellen
    pages_layouts = {}
    current_layout_id = 1
    for page in range(1, number_pages + 1):
        num_layouts = rng.randint(range_layouts_per_page[0], range_layouts_per_page[1])
        lay = list(range(current_layout_id, current_layout_id + num_layouts))
        pages_layouts[page] = lay
        current_layout_id += num_layouts
    "{page: [verf. Layouts]}"
    layouts = [lay for page in range(1, number_pages + 1) for lay in pages_layouts[page]] #Layouts als Liste
    """

    #Version 2: keine exklusiven Layouts je Seite, sondern Überlappungen möglich.
    #Schwachpunkt: Methode sorgt dafür, dass Layouts 1,2,... häufiger vorkommen, da sie zu Beginn generiert werden
    layout_pool = []
    pages_layouts = {}
    layout_counter = 1
    reuse_prob = 0.3 #passt an wie Wahrscheinlich es ist, dass ein bestehendes Layout zugewiesen wird (Dopplung)
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
    "{page: [verf. Layouts]}"
    layouts = sorted({lay for layouts in pages_layouts.values() for lay in layouts}) #Layouts als Liste


    #Erstellung der Layouts:
    unique_layouts = set()
    all_layouts = []
    while len(all_layouts) < len(layouts):
        min_width = 2
        min_height = 3 #falls mit 100 Zeilen -> height = 30 #Anpassung bei 100*6 Layout
        layout = generate_layout(min_width, min_height, rng) 
        layout_key = layout_to_tuple(layout)
        if layout_key not in unique_layouts:
            unique_layouts.add(layout_key)
            all_layouts.append(layout)
    #ins Format bringen:        
    geometry_layout_box = {
        i + 1: {
            j + 1: {"x": x, "y": y, "w": w, "h": h, "character": w * h * 40 * 10} # Character soll Anz. Zeichen in der Box beschreiben
            for j, (x, y, w, h) in enumerate(lay) #*10 nur, falls wir mit 10 Zeilen und nicht 100 rechnen! #Anpassung bei 100*6 Layout
        }
        for i, lay in enumerate(all_layouts)
    }
    box_layouts = {
        i + 1: list(range(1, len(lay) + 1))
        for i, lay in enumerate(all_layouts)
    }
    "geometry_layout_box = {Layout 1: {Box 1: {x,y,w,h,character}, Box 2: {x,y,w,h,character}}, Layout 2: {...}, ...}"
    "box_layouts = {Layout 1: [ Box 1, 2, 3], 2 : [1, 2, 3, 4, 5], ... : [...], ...}"
    

    #Zuweisung Shells zu Boxen (jede Box hat ein eigenes Shell-Sample) + Zuweisung MinMax Zeichen je Shell (abh. von Boxgr.):
    counter = 1
    shell_layout_box = {}
    shell_params = {}
    number_shells = [] #wird für Zuweisung Artikel -> Shells benötigt
    for layout_id, boxes in geometry_layout_box.items():
        shell_layout_box[layout_id] = {}
        for box_id, box_data in boxes.items():
            num_shells = rng.randint(shells_per_box_fc[0], shells_per_box_fc[1])
            character = box_data["character"]
            #Shell Ids
            shells_lst = list(range(counter, counter + num_shells))
            shell_layout_box[layout_id][box_id] = shells_lst
            #MinMax abh. je Box
            for shell_id in shells_lst:
                max_val = round(character * rng.uniform(0.5, 1.0))
                if p_type_fc == "B": #je nach Variante
                    lo = 0.75
                    hi = 0.85
                else: 
                    lo = 0.90
                    hi = 0.95
                fac = rng.uniform(lo, hi)
                min_val = round(max_val * fac)
                shell_params[shell_id] = {
                    "min": min_val,
                    "max": max_val
                }
                number_shells.append(shell_id)
            counter += num_shells
    "shell_layout_box = {Layout 1: {Box 1: [Shell 1,2,3], Box 2: [Shell 4,5]}, Layout 2: {Box 1: [Shell 6,...], ...}, ...}"
    "shell_params = {Shell 1: {min: ..., max: ...}, Shell 2 {...}, ...}"

    
    #Zuweisung Artikel -> Shells: Unterschiedliche Methoden, einfach """...""" raus/rein nehmen
    article = list(range(1, number_article + 1)) #Artikel als Liste
    shells = list(range(1, len(shell_params) + 1)) #Shells als Liste

    """
    #Methode 1: zufällig, dadurch bleiben Shells ungenutzt
    shell_article = {a: sorted(rng.sample(sh, min(rng.randint(shells_per_article_fc[0], shells_per_article_fc[1]), len(shells)))) for a in article}
    "{Artikel 1: [Shell 1, Shell 50, ...], Artikel 2: [...], ...}"
    """

    #Methode 2: in Kategorien -> deutlich mehr als 10-15 Shells je Artikel aber keine ungenutzten
    num_categories = min(len(article), 20) #min Artikel, weil sonst Kategorien ungenutzt
    categories = list(range(num_categories)) #Kategorien als Liste von 0 bis ...
    article_to_category = balanced_assignment(article, categories, rng) #damit keine Kategorie leer bleibt
    shell_to_category = balanced_assignment(shells, categories, rng)
    category_to_shells = {}
    for s, cat in shell_to_category.items():
        category_to_shells.setdefault(cat, []).append(s)
    shell_article = {art: category_to_shells.get(cat, []) for art, cat in article_to_category.items()} #####
    for a in shell_article: #Sortieren
        shell_article[a] = sorted(shell_article[a])
    shell_article = dict(sorted(shell_article.items()))
    "{Artikel 1: [Shell 1, Shell 50, ...], Artikel 2: [...], ...}"
    
    """
    #Überprüfen wie viele Shells nicht beansprucht werden:
    #nur für Methode 1 
    used_shells = set()
    for she in shell_article.values():
        used_shells.update(she)
    unused_shells = set(shells) - used_shells
    #print(len(unused_shells)/len(shells))

    #Überprüfen wie max. Anz. Shells pro Artikel
    #nur für Methode 2
    max_len = max(len(v) for v in shell_article.values())
    #print(max_len)
    """

    #Zuweisung Artikellänge
    article_lengths = {a: rng.randint(500, 20000) for a in article} 
    #eig. Max-> 24000 Zeichen je Seite, Max-> 21600 weil mind. zwei Boxen je Seite, Min -> 2400 Zeichen für kleinste Box
    #aber wir gehen bei Max ja zw. 0.5-1 * Max
    #Tests zeigen: Shell-Werte liegen zw. 900 und 17000 Zeichen
    # -> 500 und 20000 aber interpretierbar
    #Anpassung bei 100*6 Layout
    "{Artikel: [Länge d. Artikels]}"

    
    #Zuweisung Artikelpriorität: Hier als prozentualer Bereich
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


    #Nur zur Vollständigkeit:
    resorts = [1]
    article_resorts = {1: article.copy()}
    resort_page = {p: resorts.copy() for p in pages}

    
    #Output: absichtlich wie bei dir, damit richtiges Format
    return {
        "canvas": canvas,
        "pages": pages,
        "layouts": layouts,
        "article": article,
        "hulls": shells,
        "layouts_pages": pages_layouts,
        "box_layouts": box_layouts,
        "geometry_layout_box": geometry_layout_box,
        "hull_layout_box": shell_layout_box,
        "hull_article": shell_article,
        "article_length": article_lengths,
        "hull_params": shell_params,
        "article_priority": article_priority,
        "resorts": resorts,
        "article_resorts": article_resorts,
        "resort_page": resort_page,
    }

#Variationsidee der Instanzen
#Seiten: 10, 20, 30
#Artikel: Seiten * 5 +-10% (im Code noch ohne +-)

#kleine Instanz zum "ausprobieren":
pages_gl = 30
layouts_per_page_range = [2, 5] #[min,max]
shells_per_box = [5, 10] #[min,max]
article_gl = pages_gl * 5
shells_per_article = [10, 15] #[min,max] #wird nur für die zweite Methode benötigt
p_type = "B" #wenn "B" dann zw. 0.75-0.85 * Max, else 0.90-0.95
seed_gl = 42
instance = create_instance(
            pages_gl,
            layouts_per_page_range, 
            shells_per_box, 
            article_gl, 
            shells_per_article, 
            p_type, 
            seed_gl)
prefix = "Test"
i = 1
name = f"{prefix}P{pages_gl}A{article_gl}V{i}({p_type}).json"
save_instance_to_json(instance, name)
