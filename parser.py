import json
import os


def parse_json_from_file(file_name):
    with open(file_name, "r") as f:
        json_data = json.load(f)

    canvas = json_data.get("canvas", None)

    pages = list(range(1, json_data["pages"] + 1))
    articles = list(range(1, json_data["articles"] + 1))
    layouts = list(range(1, json_data["layouts"] + 1))
    sections = list(range(1, json_data["sections"] + 1))

    articles_section = {int(k): v for k, v in json_data["articles_section"].items()}
    sections_page = {int(k): v for k, v in json_data["sections_page"].items()}
    layouts_page = {int(k): v for k, v in json_data["layouts_page"].items()}
    boxes_layout = {int(k): v for k, v in json_data["boxes_layout"].items()}

    geometry_layout_box = {
        int(layout): {int(box): geom for box, geom in boxes.items()}
        for layout, boxes in json_data.get("geometry_layout_box", {}).items()
    }

    shells_layout_box = {
        int(layout): {int(box): shells for box, shells in boxes.items()}
        for layout, boxes in json_data["shells_layout_box"].items()
    }

    shells_article = {int(k): v for k, v in json_data["shells_article"].items()}
    length_article = {int(k): v for k, v in json_data["length_article"].items()}
    priority_article = {int(k): v for k, v in json_data["priority_article"].items()}
    params_shell = {int(k): v for k, v in json_data["params_shell"].items()}

    return (
        pages, articles, layouts, sections,
        articles_section, sections_page, layouts_page,
        boxes_layout, geometry_layout_box,
        shells_layout_box, shells_article, length_article,
        params_shell, priority_article
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    name = os.path.join(base_dir, "Instances", "Diss", "Medium", "Instance_10_60_1.json")

    (
        pages, articles, layouts, sections,
        articles_section, sections_page, layouts_page,
        boxes_layout, geometry_layout_box,
        shells_layout_box, shells_article, length_article,
        params_shell, priority_article
    ) = parse_json_from_file(name)

    liste = []
    for i in params_shell:
        percentage = (params_shell[i]["max"] - params_shell[i]["min"]) / params_shell[i]["max"]
        liste.append(percentage)

    print("canvas:", canvas)
    print("layouts mit geometry:", list(geometry_layout_box.keys())[:10])
    print(shells_layout_box.keys())
    print(max(liste))
    print(min(liste))
    print(sum(liste) / len(liste))