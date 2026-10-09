import cv2
import numpy as np
from napari_builtins._measure_shapes import ellipse_area, polygon_area, rectangle_area
from track.modelling.onnx_inference import predict_mask
from skimage.filters import threshold_isodata
from skimage.measure import label
from skimage.morphology import closing, opening


def measure_area(shape_data_list, shape_type_list) -> float:
    """calculates area of shapes"""
    vertices = shape_data_list[0]
    shape_type = shape_type_list[0]
    if shape_type == "polygon":
        area = polygon_area(vertices)
    elif shape_type == "rectangle":
        area = rectangle_area(vertices)
    elif shape_type == "ellipse":
        area = ellipse_area(vertices)
    elif shape_type == "path" or shape_type == "line":
        area = 0
    else:
        raise ValueError("Unsupported shape type")
    return area


def density_calc(num_of_tracks: np.intp | int, area: float, scale: float) -> float:
    """Calculates tracks density"""
    if area != 0:
        tracks_density_by_pixel = num_of_tracks / area
        tracks_density = tracks_density_by_pixel * scale**2
    else:
        tracks_density = 0
    return tracks_density


def frame_postproc(video_buffer: np.ndarray) -> np.ndarray:
    """
    Postprocessing of livestream frame from pymmcore-plus
    """
    height, width = video_buffer.shape
    if video_buffer.dtype == np.uint32:
        pix_type = np.uint8
    elif video_buffer.dtype == np.uint64:
        pix_type = np.uint16
    else:
        raise ValueError(
            f"Неизвестный тип пикселей для toupcam: {video_buffer.dtype!r}. "
            f"Допустимые: np.uint32, np.uint64."
        )

    # 1. Раскладываем матрицу uint32 на 4 отдельных байта по третьей оси.
    # Теперь массив имеет форму [Height, Width, 4] и тип uint8.
    bytes_array = video_buffer.view(pix_type).reshape(height, width, 4)

    # 2. Выделяем память под итоговый uint8 массив [Height, Width, 3]
    rgb_img = np.zeros((height, width, 3), dtype=pix_type)

    # 3. Достаем каналы из байтов.
    # В упакованном uint32 (BGRA) байты идут строго друг за другом.
    # Сразу переводим в uint8 и масштабируем (<< 8), чтобы получить диапазон 0-65535.

    # Предполагаем стандартный для ToupCam порядок BGRA (байт 0 = B, байт 1 = G, байт 2 = R):
    b = bytes_array[:, :, 0].astype(pix_type)
    g = bytes_array[:, :, 1].astype(pix_type)
    r = bytes_array[:, :, 2].astype(pix_type)

    rgb_img[:, :, 0] = r  # Red
    rgb_img[:, :, 1] = g  # Green
    rgb_img[:, :, 2] = b  # Blue
    return rgb_img


def translate_layer(layer):
    """Calculates new translation values for some layer based on given layer shape"""
    if layer is not None:
        # Move the layer along the X-axis
        current_translation = list(layer.translate)
        current_translation[-1] = layer.data.shape[
            1
        ]  # Modifying the last dimension (X)
        return current_translation
