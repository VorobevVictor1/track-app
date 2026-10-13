import numpy as np
# from napari_builtins._measure_shapes import ellipse_area, polygon_area, rectangle_area


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
    """calculates area of shapes"""
    if shape_type == "polygon":
        area = _polygon_area(shape_verticies)
    elif shape_type == "rectangle":
        area = _rectangle_area(shape_verticies)
    elif shape_type == "ellipse":
        area = _ellipse_area(shape_verticies)
    else:
        raise ValueError("Unsupported ROI shape type")
    return area


def shapes_to_mask_and_area(
    shapes_layer, mask_shape: np.ndarray
) -> tuple[np.ndarray | None, float]:
    if shapes_layer is None or shapes_layer.nshapes == 0:
        return None, 0.0

    # TODO: implement several ROIs and excluding ROI
    if shapes_layer.nshapes > 1:
        raise ValueError("Currently only one ROI is supported. ")

    mask = shapes_layer.to_masks(mask_shape=mask_shape)[0]

    # takes first and only values from returned lists
    area_px = measure_shape_area(shapes_layer.data[0], shapes_layer.shape_type[0])

    return mask, area_px
