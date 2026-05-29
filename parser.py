
import json
import numpy as np
import os

def parse_json_from_file(file_name):
    with open(file_name, "r") as f:
        json_data = json.load(f)

    print(json_data["sections"])

    canvas = json_data.get("canvas", None)

    pages = json_data["pages"]
    article = json_data["article"]
    layouts = json_data["layouts"]
    sections = json_data["sections"]

    article_sections = {int(k): v for k, v in json_data["article_sections"].items()}
    sections_page = {int(k): v for k, v in json_data["sections_page"].items()}
    layouts_pages = {int(k): v for k, v in json_data["layouts_pages"].items()}
    box_layouts = {int(k): v for k, v in json_data["box_layouts"].items()}

    geometry_layout_box = {
        int(layout): {int(box): geom for box, geom in boxes.items()}
        for layout, boxes in json_data.get("geometry_layout_box", {}).items()
    }

    shells_layout_box = {
        int(layout): {int(box): hulls for box, hulls in boxes.items()}
        for layout, boxes in json_data["shells_layout_box"].items()
    }

    shells_article = {int(k): v for k, v in json_data["shells_article"].items()}
    article_length = {int(k): v for k, v in json_data["article_length"].items()}
    article_priority = {int(k): v for k, v in json_data["article_priority"].items()}
    shell_params = {int(k): v for k, v in json_data["shell_params"].items()}

    return (
        pages, article, layouts, sections,
        article_sections, sections_page, layouts_pages,
        box_layouts, geometry_layout_box,
        shells_layout_box, shells_article, article_length,
        shell_params, article_priority
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    name = os.path.join(base_dir, "Instances", "Diss", "Medium", "Instance_10_60_1.json")

    (
     pages, article, layouts, sections,
     article_sections, sections_page, layouts_pages,
     box_layouts, geometry_layout_box,
     shells_layout_box, shells_article, article_length,
     shell_params, article_priority) = parse_json_from_file(name)

    liste = []
    for i in shell_params:
        percentage = (shell_params[i]["max"] - shell_params[i]["min"]) / shell_params[i]["max"]
        liste.append(percentage)

    print("canvas:", canvas)
    print("layouts mit geometry:", list(geometry_layout_box.keys())[:10])
    print(shells_layout_box.keys())
    print(max(liste))
    print(min(liste))
    print(sum(liste) / len(liste))
