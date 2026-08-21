import json
import os


def parse_json_from_file(file_name):
    with open(file_name, "r") as f:
        json_data = json.load(f)


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

    # NEU: Einlesen der vertical_chains
    layout_vertical_chains = {
        int(layout): chains 
        for layout, chains in json_data.get("layout_vertical_chains", {}).items()
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
        params_shell, priority_article, layout_vertical_chains
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    name = os.path.join(base_dir, "Instances", "Diss", "Medium", "Instance_10_60_1.json")

    (
        pages, articles, layouts, sections,
        articles_section, sections_page, layouts_page,
        boxes_layout, geometry_layout_box,
        shells_layout_box, shells_article, length_article,
        params_shell, priority_article, layout_vertical_chains
    ) = parse_json_from_file(name)
