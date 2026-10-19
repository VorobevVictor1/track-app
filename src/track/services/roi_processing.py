import numpy as np
from ..data_models import ROI, ROIType
from collections.abc import Callable
from cv2 import fillPoly, minAreaRect, ellipse, drawMarker, MARKER_CROSS
# from napari_builtins._measure_shapes import ellipse_area, polygon_area, rectangle_area

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari.layers


def _polygon_area(vertices) -> float:
    coords = np.asarray(vertices)
    n, _dims = coords.shape

    if n < 3:
        return 0.0

    area = 0.0

    # use a reference vertex to make the polygon out of a "triangle fan"
    ref = coords[0]
    for i in range(1, n - 1):
        u = coords[i] - ref
        v = coords[i + 1] - ref
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        cross_sq = uu * vv - uv**2
        triangle_area = 0.5 * np.sqrt(max(cross_sq, 0))
        area += triangle_area

    return area


def _rectangle_area(vertices) -> float:
    return np.linalg.norm(vertices[1] - vertices[0]) * np.linalg.norm(
        vertices[2] - vertices[1]
    )


def _ellipse_area(vertices) -> float:
    return _rectangle_area(vertices) * np.pi / 4


def measure_shape_area(shape_verticies: np.ndarray, shape_type: str) -> float:
    float_vertices = shape_verticies.astype(dtype=float, casting="same_value")
    """calculates area of shapes"""
    if shape_type == "polygon":
        area = _polygon_area(float_vertices)
    elif shape_type == "rectangle":
        area = _rectangle_area(float_vertices)
    elif shape_type == "ellipse":
        area = _ellipse_area(float_vertices)
    else:
        raise ValueError("Unsupported ROI shape type")
    return area


def _polygon_to_mask(verticies: np.ndarray, where_to_draw: np.ndarray) -> np.ndarray:
    reshaped_vertices = verticies.reshape((-1, 1, 2))
    fillPoly(where_to_draw, [reshaped_vertices], 1)
    return where_to_draw


def _ellipse_to_mask(verticies: np.ndarray, where_to_draw: np.ndarray) -> np.ndarray:
    box = minAreaRect(verticies)
    ellipse(where_to_draw, box, color=1, thickness=-1)
    return where_to_draw


def shape_to_mask(
    shape_verticies: np.ndarray, shape_type: str, where_to_draw: np.ndarray
) -> np.ndarray:
    if shape_type == "polygon" or shape_type == "rectangle":
        where_to_draw = _polygon_to_mask(shape_verticies, where_to_draw)
    elif shape_type == "ellipse":
        where_to_draw = _ellipse_to_mask(shape_verticies, where_to_draw)
    else:
        raise ValueError("Unsupported ROI shape type")
    return where_to_draw


def draw_roi_mask(
    roi_list: list[ROI], mask_shape: np.ndarray | None
) -> np.ndarray | None:
    if mask_shape is None:
        return None

    include_mask = np.zeros(mask_shape, dtype=np.uint8)
    exclude_mask = np.zeros(mask_shape, dtype=np.uint8)

    for roi in roi_list:
        if roi.roi_type == ROIType.INCLUDE:
            include_mask = shape_to_mask(
                shape_verticies=np.flip(roi.vertices, 1),
                shape_type=roi.shape_type,
                where_to_draw=include_mask,
            )
        elif roi.roi_type == ROIType.EXCLUDE:
            exclude_mask = shape_to_mask(
                shape_verticies=np.flip(roi.vertices, 1),
                shape_type=roi.shape_type,
                where_to_draw=exclude_mask,
            )
        else:
            raise ValueError("Unsupported ROI type")
    """
    left = include mask value, right = exclude mask value
    rule:
    1 and 1 -> 0 (1 - 1 * 1 = 0)
    1 and 0 -> 1 (1 - 1 * 0 = 1)
    0 and 1 -> 0 (0 - 0 * 1 = 0)
    0 and 0 -> 0 (0 - 0 * 0 = 0)
    """
    return include_mask - include_mask * exclude_mask


def exctract_rois_data(shapes_layer, target_world_to_data: Callable) -> list[ROI]:
    """
    Returns rois vertices in target image data coordinates ready to draw, shape type and roi type(include or exclude)

    Args:
        shapes_layer (_type_): napari shapes layer

    Returns:
        list[ROI]: list with ROIs information
    """
    shapes_list = shapes_layer.data
    shape_types_list = shapes_layer.shape_type
    roi_list = []

    for shape, type in zip(shapes_list, shape_types_list):
        vertices_world = np.apply_along_axis(
            shapes_layer.data_to_world, axis=1, arr=shape
        )
        vertices_target_data = np.apply_along_axis(
            target_world_to_data, axis=1, arr=vertices_world
        )
        vertices_rounded = np.rint(vertices_target_data)
        roi_list.append(
            ROI(
                vertices=vertices_rounded.astype(dtype=np.int32, casting="same_value"),
                shape_type=type,
            )
        )
    return roi_list


def shapes_to_mask_and_area(
    shapes_layer: "napari.layers.Shapes",
    mask_shape: np.ndarray,
    target_world_to_data: Callable,
) -> tuple[np.ndarray | None, float]:
    if shapes_layer is None or shapes_layer.nshapes == 0:
        return None, 0.0

    # TODO: implement several ROIs and excluding ROI
    if shapes_layer.nshapes > 1:
        raise ValueError("Currently only one ROI is supported. ")

    roi_list = exctract_rois_data(shapes_layer, target_world_to_data)
    mask = draw_roi_mask(roi_list, mask_shape)

    # we use vertices in target image data coordinates because we use those pixels size
    # takes first and only values from returned lists
    area_px = measure_shape_area(roi_list[0].vertices, roi_list[0].shape_type)

    return mask, area_px


def points_to_mask(
    points_layer: "napari.layers.Points",
    mask_shape: np.ndarray,
    target_world_to_data: Callable,
) -> np.ndarray:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    points_data = points_layer.data
    points_world = np.apply_along_axis(
        points_layer.data_to_world, axis=1, arr=points_data
    )
    points_target_data = np.apply_along_axis(
        target_world_to_data, axis=1, arr=points_world
    )
    points_rounded = np.rint(points_target_data)
    points_int = points_rounded.astype(dtype=np.int32, casting="same_value")
    # меняем порядок x и y для отрисовки через opencv
    point_sizes = points_layer.size
    for point, size in zip(np.flip(points_int, 1), point_sizes):
        drawMarker(
            img=mask,
            position=point,
            color=1,
            markerType=MARKER_CROSS,
            markerSize=size,
            thickness=6,
        )
    return mask
