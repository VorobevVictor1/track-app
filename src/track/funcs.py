import numpy as np
#from napari_builtins._measure_shapes import ellipse_area, polygon_area, rectangle_area


def _polygon_area(vertices) -> np.float64:
    coords = np.asarray(vertices)
    n, _dims = coords.shape

    if n < 3:
        return 0

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


def _rectangle_area(vertices) -> np.float64:
    return np.linalg.norm(vertices[1] - vertices[0]) * np.linalg.norm(
        vertices[2] - vertices[1]
    )


def _ellipse_area(vertices) -> np.float64:
    return _rectangle_area(vertices) * np.pi / 4


def measure_area(shape_data_list, shape_type_list) -> np.float64:
    """calculates area of shapes"""
    vertices = shape_data_list[0]
    shape_type = shape_type_list[0]
    if shape_type == "polygon":
        area = _polygon_area(vertices)
    elif shape_type == "rectangle":
        area = _rectangle_area(vertices)
    elif shape_type == "ellipse":
        area = _ellipse_area(vertices)
    else:
        raise ValueError("Unsupported ROI shape type")
    return area


def density_calc(num_of_tracks: np.intp | int, area: np.float64, scale: float) -> np.float64:
    """Calculates tracks density"""
    if area != 0:
        tracks_density_by_pixel = num_of_tracks / area
        tracks_density = tracks_density_by_pixel * scale**2
    else:
        tracks_density = 0
    return tracks_density
