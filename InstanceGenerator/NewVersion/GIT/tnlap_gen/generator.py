"""Parameterized instance generator for the Template-based Newspaper Layout
Assignment Problem (TNLAP)"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
COLS: int = 6
ROWS: int = 100
CHARS_PER_CELL: int = 40 

# Box geometry bounds used by the recursive partitioning
MIN_BOX_WIDTH: int = 2
MIN_BOX_HEIGHT: int = 30

Rect = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    """Configuration of the hard instance generator."""

    shell_type: str = "A"

    layouts_per_page: Tuple[int, int] = (5, 7)
    boxes_per_layout: Tuple[int, int] = (3, 5)
    shells_per_box: Tuple[int, int] = (6, 15)
    max_categories: int = 20

    article_length: Tuple[int, int] = (500, 15_000)
    shell_max_fraction: Tuple[float, float] = (0.75, 1.0)

    # Lower-bound fraction f, drawn per shell, by instance type.
    shell_min_fraction: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"A": (0.96, 0.99), "B": (0.92, 0.96)}
    )

    # Priority tier fractions, drawn once per instance.
    priority_a_fraction: Tuple[float, float] = (0.10, 0.20)
    priority_b_fraction: Tuple[float, float] = (0.30, 0.40)


    def __post_init__(self) -> None:
        if self.shell_type not in self.shell_min_fraction:
            raise ValueError(
                f"shell_type must be one of "
                f"{sorted(self.shell_min_fraction)}, got {self.shell_type!r}"
            )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def split_rect(
    rect: Rect, min_width: int, min_height: int, rng: random.Random
) -> Tuple[Rect, Optional[Rect]]:
    """Split a  page into rectangles."""
    x, y, w, h = rect
    can_split_vertical = w > 2 * min_width
    can_split_horizontal = h > 2 * min_height

    if can_split_vertical and can_split_horizontal:
        split_vertical = rng.choice([True, False])
    elif can_split_vertical:
        split_vertical = True
    elif can_split_horizontal:
        split_vertical = False
    else:
        return rect, None

    if split_vertical:
        cut = rng.randint(min_width, w - min_width)
        return (x, y, cut, h), (x + cut, y, w - cut, h)
    cut = rng.randint(min_height, h - min_height)
    return (x, y, w, cut), (x, y + cut, w, h - cut)


def generate_layout(config: GeneratorConfig, rng: random.Random) -> List[Rect]:
    """Generate a single master layout as a partition of the full grid."""
    min_boxes, max_boxes = config.boxes_per_layout
    target_boxes = rng.randint(min_boxes, max_boxes)

    rects: List[Rect] = [(0, 0, COLS, ROWS)]
    attempts = 0
    max_attempts = 200
    while len(rects) < target_boxes and attempts < max_attempts:
        attempts += 1
        idx = rng.randrange(len(rects))
        rect = rects.pop(idx)
        r1, r2 = split_rect(rect, MIN_BOX_WIDTH, MIN_BOX_HEIGHT, rng)
        if r2 is None:
            rects.append(r1)
            continue
        rects.extend([r1, r2])
    return rects


def _layout_key(layout: List[Rect]) -> Tuple[Rect, ...]:
    """Canonical, hashable key for deduplicating layouts."""
    return tuple(sorted(layout))


def _reading_order(layout: List[Rect]) -> List[Rect]:
    """Sort boxes by the row in which they end, then left-to-right."""
    return sorted(layout, key=lambda r: (r[1] + r[3], r[0]))


def box_capacity(width: int, height: int) -> int:
    """Upper bound on the body-text characters a box can hold."""
    return width * height * CHARS_PER_CELL

#####
def _get_vertical_chains(geometry: Dict[int, Dict[int, Dict[str, int]]]) -> Dict[int, List[Tuple[int, ...]]]:
    """Ermittelt alle maximalen vertikalen Box-Ketten für jedes Layout und gibt diese als Dictionary zurück."""
    layout_vertical_chains = {}
    
    for layout_id, boxes in geometry.items():
        box_ids = list(boxes.keys())
        
        adj = {b_id: [] for b_id in box_ids}
        in_degree = {b_id: 0 for b_id in box_ids}
        
        for i in range(len(box_ids)):
            for j in range(len(box_ids)):
                if i == j:
                    continue
                    
                b1_id = box_ids[i]
                b2_id = box_ids[j]
                b1 = boxes[b1_id]
                b2 = boxes[b2_id]
                
                # Bedingung 1: B1 ist exakt über B2
                if b1["y"] + b1["h"] == b2["y"]:
                    # Bedingung 2: Horizontaler Überlapp
                    overlap_left = max(b1["x"], b2["x"])
                    overlap_right = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
                    
                    if overlap_left < overlap_right:
                        adj[b1_id].append(b2_id)
                        in_degree[b2_id] += 1
                        
        roots = [b_id for b_id in box_ids if in_degree[b_id] == 0 and len(adj[b_id]) > 0]
        chains = []
        
        def dfs(current_node, current_path):
            if not adj[current_node]:
                if len(current_path) > 1:
                    chains.append(tuple(current_path))
                return
            for child in adj[current_node]:
                dfs(child, current_path + [child])
                
        for root in roots:
            dfs(root, [root])
            
        layout_vertical_chains[layout_id] = sorted(chains)
        
    return layout_vertical_chains
####


# ---------------------------------------------------------------------------
# Assignment helpers
# ---------------------------------------------------------------------------
def balanced_assignment(
    items: List[int], categories: List[int], rng: random.Random
) -> Dict[int, int]:
    """Distribute items evenly across categories via shuffled round-robin."""
    shuffled = items[:]
    rng.shuffle(shuffled)
    return {item: categories[i % len(categories)] for i, item in enumerate(shuffled)}


# ---------------------------------------------------------------------------
# Instance construction steps
# ---------------------------------------------------------------------------
def _assign_layouts_to_pages(
    pages: List[int], config: GeneratorConfig, rng: random.Random
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Assign unique candidate master-layout IDs to each page."""
    lo, hi = config.layouts_per_page
    pages_layouts: Dict[int, List[int]] = {}
    next_id = 1

    for page in pages:
        n_layouts = rng.randint(lo, hi)

        selected = list(range(next_id, next_id + n_layouts))
        pages_layouts[page] = selected

        next_id += n_layouts

    layouts = list(range(1, next_id))

    return pages_layouts, layouts


def _build_layout_geometry(
    n_layouts: int, config: GeneratorConfig, rng: random.Random
) -> Dict[int, Dict[int, Dict[str, int]]]:
    """Generate ``n_layouts`` distinct layouts and their box geometries."""
    seen: set = set()
    layouts: List[List[Rect]] = []
    while len(layouts) < n_layouts:
        layout = generate_layout(config, rng)
        key = _layout_key(layout)
        if key not in seen:
            seen.add(key)
            layouts.append(layout)

    return {
        layout_id + 1: {
            box_id + 1: {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "max_character": box_capacity(w, h),
            }
            for box_id, (x, y, w, h) in enumerate(_reading_order(layout))
        }
        for layout_id, layout in enumerate(layouts)
    }


def _build_shells(
    geometry: Dict[int, Dict[int, Dict[str, int]]],
    config: GeneratorConfig,
    rng: random.Random,
) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[int, Dict[str, int]]]:
    """Create shells per box and draw their character intervals."""
    lo_n, hi_n = config.shells_per_box
    frac_lo, frac_hi = config.shell_max_fraction
    min_lo, min_hi = config.shell_min_fraction[config.shell_type]

    shells_layout_box: Dict[int, Dict[int, List[int]]] = {}
    shell_params: Dict[int, Dict[str, int]] = {}
    next_id = 1

    for layout_id, boxes in geometry.items():
        shells_layout_box[layout_id] = {}
        for box_id, box in boxes.items():
            n_shells = rng.randint(lo_n, hi_n)
            shell_ids = list(range(next_id, next_id + n_shells))
            shells_layout_box[layout_id][box_id] = shell_ids
            next_id += n_shells

            for shell_id in shell_ids:
                max_val = round(box["max_character"] * rng.uniform(frac_lo, frac_hi))
                min_val = round(max_val * rng.uniform(min_lo, min_hi))
                shell_params[shell_id] = {"min": min_val, "max": max_val}

    return shells_layout_box, shell_params


def _assign_shells_to_articles(
    articles: List[int],
    shells: List[int],
    config: GeneratorConfig,
    rng: random.Random,
) -> Dict[int, List[int]]:
    """Assign each article the shells sharing its category.

    Articles and shells are distributed evenly over a common set of
    categories, so every shell belongs to the candidate set of at least
    one article as long as there are at least as many shells as
    categories.
    """
    n_categories = min(len(articles), config.max_categories)
    categories = list(range(n_categories))

    article_category = balanced_assignment(articles, categories, rng)
    shell_category = balanced_assignment(shells, categories, rng)

    category_shells: Dict[int, List[int]] = {}
    for shell, cat in shell_category.items():
        category_shells.setdefault(cat, []).append(shell)

    return {
        art: sorted(category_shells.get(cat, []))
        for art, cat in sorted(article_category.items())
    }


def _assign_priorities(
    articles: List[int], config: GeneratorConfig, rng: random.Random
) -> Dict[int, str]:
    """Partition articles into disjoint priority tiers A, B, C."""
    n = len(articles)
    n_a = max(1, round(rng.uniform(*config.priority_a_fraction) * n))
    n_b = max(1, round(rng.uniform(*config.priority_b_fraction) * n))

    remaining = articles.copy()
    tier_a = set(rng.sample(remaining, n_a))
    remaining = [a for a in remaining if a not in tier_a]
    tier_b = set(rng.sample(remaining, n_b))

    return {
        a: ("A" if a in tier_a else "B" if a in tier_b else "C") for a in articles
    }


def _assign_sections(
    pages: List[int],
    articles: List[int],
) -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[int]]]:
    """Assign all pages to section 1 and all articles to section 1."""

    sections = [1]

    article_sections = {
        1: articles.copy()
    }

    sections_page = {
        page: [1] for page in pages
    }

    return sections, article_sections, sections_page


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_instance(
    n_pages: int,
    n_articles: int,
    shell_type: str = "A",
    seed: Optional[int] = None,
    config: Optional[GeneratorConfig] = None,
) -> Dict[str, object]:
    """Generate a single TNLAP instance.

    Args:
        n_pages: Number of pages in the edition.
        n_articles: Number of articles in the content pool.
        shell_type: ``"A"`` (very tight) or ``"B"`` (tight). Ignored if an
            explicit ``config`` is supplied.
        seed: Seed for the random number generator (for reproducibility).
        config: Optional explicit configuration.

    Returns:
        A dictionary describing the instance (see README).
    """
    if config is None:
        config = GeneratorConfig(shell_type=shell_type)
    rng = random.Random(seed)

    pages = list(range(1, n_pages + 1))
    articles = list(range(1, n_articles + 1))

    pages_layouts, layouts = _assign_layouts_to_pages(pages, config, rng)
    geometry = _build_layout_geometry(len(layouts), config, rng)
    ####
    layout_vertical_chains = _get_vertical_chains(geometry)
    ####
    box_layouts = {lid: sorted(boxes) for lid, boxes in geometry.items()}

    shells_layout_box, shell_params = _build_shells(geometry, config, rng)
    shells = sorted(shell_params)

    shells_article = _assign_shells_to_articles(articles, shells, config, rng)

    article_length = {a: rng.randint(*config.article_length) for a in articles}
    article_priority = _assign_priorities(articles, config, rng)
    sections, article_sections, sections_page = _assign_sections(
        pages, articles
    )

    return {
        "pages": len(pages),
        "layouts": len(layouts),
        "articles": len(articles),
        "shells": len(shells),
        "layouts_page": pages_layouts,
        "boxes_layout": box_layouts,
        "geometry_layout_box": geometry,
        "shells_layout_box": shells_layout_box,
        "shells_article": shells_article,
        "length_article": article_length,
        "params_shell": shell_params,
        "priority_article": article_priority,
        "sections": len(sections),
        "articles_section": article_sections,
        "sections_page": sections_page,
        #"layout_vertical_chains": layout_vertical_chains,
    }


def save_instance(instance: Dict[str, object], path: str) -> None:
    """Write an instance to a JSON file (UTF-8, indented)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=4, ensure_ascii=False)
