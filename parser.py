
import json
import numpy as np
import os

def parse_json_from_file(file_name):
    with open(file_name, "r") as f:
        json_data = json.load(f)

    print(json_data["resorts"])

    canvas = json_data.get("canvas", None)

    pages = json_data["pages"]
    article = json_data["article"]
    layouts = json_data["layouts"]
    resorts = json_data["resorts"]

    article_resorts = {int(k): v for k, v in json_data["article_resorts"].items()}
    resort_page = {int(k): v for k, v in json_data["resort_page"].items()}
    layouts_pages = {int(k): v for k, v in json_data["layouts_pages"].items()}
    box_layouts = {int(k): v for k, v in json_data["box_layouts"].items()}

    geometry_layout_box = {
        int(layout): {int(box): geom for box, geom in boxes.items()}
        for layout, boxes in json_data.get("geometry_layout_box", {}).items()
    }

    hull_layout_box = {
        int(layout): {int(box): hulls for box, hulls in boxes.items()}
        for layout, boxes in json_data["hull_layout_box"].items()
    }

    hull_article = {int(k): v for k, v in json_data["hull_article"].items()}
    article_length = {int(k): v for k, v in json_data["article_length"].items()}
    article_priority = {int(k): v for k, v in json_data["article_priority"].items()}
    hull_params = {int(k): v for k, v in json_data["hull_params"].items()}

    return (
        canvas,
        pages, article, layouts, resorts,
        article_resorts, resort_page, layouts_pages,
        box_layouts, geometry_layout_box,
        hull_layout_box, hull_article, article_length,
        hull_params, article_priority
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    name = os.path.join(base_dir, "Instances", "Diss", "Medium", "Instance_10_60_1.json")

    (canvas,
     pages, article, layouts, resorts,
     article_resorts, resort_page, layouts_pages,
     box_layouts, geometry_layout_box,
     hull_layout_box, hull_article, article_length,
     hull_params, article_priority) = parse_json_from_file(name)

    liste = []
    for i in hull_params:
        percentage = (hull_params[i]["max"] - hull_params[i]["min"]) / hull_params[i]["max"]
        liste.append(percentage)

    print("canvas:", canvas)
    print("layouts mit geometry:", list(geometry_layout_box.keys())[:10])
    print(hull_layout_box.keys())
    print(max(liste))
    print(min(liste))
    print(sum(liste) / len(liste))
