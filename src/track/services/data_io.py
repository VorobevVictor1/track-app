from cv2 import COLOR_RGB2BGR, cvtColor, imencode
from typing import TYPE_CHECKING
import csv
import numpy as np
import tomli_w
import napari.layers

if TYPE_CHECKING:
    from pathlib import Path

HEADER = ["grain number", "number of tracks", "Area, um^2", "Density, 1/um^2"]


def save_img(data: "np.ndarray", path: "Path", extension: str):
    res, buffer = imencode(extension, data)
    if res:
        with open(path, "wb") as f:
            f.write(buffer)
    else:
        raise Exception("Unable to save image")


def gen_base_meta_dict(layer: napari.layers.Layer) -> dict:
    name = layer.name
    affine = layer.affine.affine_matrix.tolist()
    scale = layer.scale.tolist()
    translate = layer.translate.tolist()
    opacity = layer.opacity
    blending = layer.blending
    projection_mode = layer.projection_mode
    layer_type = layer._type_string
    return {
        "name": name,
        "affine": affine,
        "scale": scale,
        "translate": translate,
        "opacity": opacity,
        "blending": blending,
        "projection_mode": projection_mode,
        "layer_type": layer_type,
    }


def save_layer_with_img(
    layer: napari.layers.Image | napari.layers.Labels, source_dir_path: "Path"
):
    meta_dict = gen_base_meta_dict(layer)
    layer_data = np.asarray(layer.data)
    file_ext = ".png"
    path = source_dir_path / (meta_dict["name"] + file_ext)
    if isinstance(layer, napari.layers.Image):
        data = cvtColor(layer_data, COLOR_RGB2BGR)
    else:
        data = layer_data.astype(np.uint8)
    save_img(data, path, file_ext)

    with open(source_dir_path / (meta_dict["name"] + "_meta.toml"), "wb") as f:
        tomli_w.dump(meta_dict, f)


def save_layer_with_shapes(layer: napari.layers.Shapes, source_dir_path: "Path"):
    meta_dict = gen_base_meta_dict(layer)
    meta_dict["face_color"] = layer.face_color.tolist()
    meta_dict["edge_color"] = layer.edge_color.tolist()
    meta_dict["edge_width"] = layer.edge_width
    path = source_dir_path / (meta_dict["name"] + ".csv")
    layer.save(str(path))

    with open(source_dir_path / (meta_dict["name"] + "_meta.toml"), "wb") as f:
        tomli_w.dump(meta_dict, f)


def save_layer_with_points(layer: napari.layers.Points, source_dir_path: "Path"):
    meta_dict = gen_base_meta_dict(layer)
    meta_dict["face_color"] = layer.face_color.tolist()
    meta_dict["border_color"] = layer.border_color.tolist()
    meta_dict["size"] = layer.size.tolist()
    meta_dict["symbol"] = layer.symbol.tolist()

    path = source_dir_path / (meta_dict["name"] + ".csv")
    layer.save(str(path))

    with open(source_dir_path / (meta_dict["name"] + "_meta.toml"), "wb") as f:
        tomli_w.dump(meta_dict, f)


def save_source_data(layers_list: list["napari.layers.Layer"], source_dir_path: "Path"):
    if not source_dir_path.exists():
        source_dir_path.mkdir()
    for layer in layers_list:
        if isinstance(layer, napari.layers.Image) or isinstance(
            layer, napari.layers.Labels
        ):
            save_layer_with_img(layer, source_dir_path)
        elif isinstance(layer, napari.layers.Shapes):
            save_layer_with_shapes(layer, source_dir_path)
        elif isinstance(layer, napari.layers.Points):
            save_layer_with_points(layer, source_dir_path)


def uppend_result_table(
    grain_number: str, tracks_num: int, area: float, density: float, table_path: "Path"
):
    dict_to_write = {
        HEADER[0]: grain_number,
        HEADER[1]: tracks_num,
        HEADER[2]: area,
        HEADER[3]: density,
    }
    if not table_path.is_file():
        with open(table_path, "w", newline="") as f:
            writer = csv.DictWriter(f, HEADER, delimiter=";")
            writer.writeheader()
            writer.writerow(dict_to_write)
    else:
        with open(table_path, "a", newline="") as f:
            writer = csv.DictWriter(f, HEADER, delimiter=";")
            writer.writerow(dict_to_write)


def save_screenshot(screenshot_array, screenshots_dir: "Path"):
    save_img(cvtColor(screenshot_array, COLOR_RGB2BGR), screenshots_dir, ".jpg")
