import cv2
import numpy as np
from napari_builtins._measure_shapes import ellipse_area, polygon_area, rectangle_area
from track.modelling.onnx_inference import predict_mask
from skimage.filters import threshold_isodata
from skimage.measure import label
from skimage.morphology import closing, opening


def instance_segment(
    img_transmitted: np.ndarray,
    img_reflected: np.ndarray,
    mask: np.ndarray,
    sigma: float = 25,
    gray_level: int = 128,
) -> np.ndarray:
    """
    Performs full instance segmentation pipeline with coincidence mapping algorithm.

    Args:
        img_transmitted (np.ndarray): transmitted grain image RGB or grayscale
        img_reflected (np.ndarray): reflected grain image of the same type as img_transmitted
        mask (np.ndarray): ROI mask for segmentation postprocessing and track density calculation
        sigma (float, optional): Dispersion for gaussian kernel. Defaults to 25.
        gray_level (int, optional): Gray level shift. Defaults to 128.

    Returns:
        np.ndarray: instance segmentation labels
    """
    # TODO fix check of size equivalence
    if img_transmitted.shape[:2] != img_reflected.shape[:2]:
        return None

    # TODO auto selection of gray level
    # returns high pass img with flat background level
    img_transmitted_hp = _background_correction(img_transmitted, sigma, gray_level)
    img_reflected_hp = _background_correction(img_reflected, sigma, gray_level)

    transmitted_features = _features_exctraction(img_transmitted_hp)
    reflected_features = _features_exctraction(img_reflected_hp)

    # coincedence mapping
    res_binary = transmitted_features * reflected_features
    # ROI mask application
    if mask is not None:
        res_binary = res_binary * mask

    filtered_binary = _morph_filtering(res_binary)

    labeled = label(filtered_binary).astype(np.uint16)

    return labeled


def segm_with_nn(
    img_transmitted: np.ndarray,
    img_reflected: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Performs full instance segmentation pipeline with umap nn.

    Args:
        img_transmitted (np.ndarray): transmitted grain image RGB or grayscale
        img_reflected (np.ndarray): reflected grain image of the same type as img_transmitted
        mask (np.ndarray): ROI mask for segmentation postprocessing and track density calculation
        threshold (float): logit threshold for segm postprocessing

    Returns:
        np.ndarray: instance segmentation labels
    """
    res_binary = predict_mask(img_reflected, img_transmitted, threshold)

    # ROI mask application
    if mask is not None:
        res_binary = res_binary * mask

    labeled = label(res_binary).astype(np.uint16)

    return labeled


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


def _background_correction(
    img: np.ndarray, sigma: float, gray_level: int
) -> np.ndarray:
    """mitigates background brightness fluctuations"""
    if len(img.shape) > 2:
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = img.astype(np.uint8)

    # Create a low-pass filtered version
    low_pass = cv2.GaussianBlur(grayscale, (0, 0), sigma)

    # Subtract low frequencies from the original image
    # We add 128 to shift the neutral gray level to the center
    high_pass = cv2.addWeighted(grayscale, 1.0, low_pass, -1.0, gray_level)

    return high_pass


def _features_exctraction(img: np.ndarray) -> np.ndarray:
    """performs segmentation by thresholding prepared image"""
    # calculates negative feature map: features pixels = 1 and background = 0
    # _, feature_map = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)
    iso_thresh = threshold_isodata(img)
    _, feature_map = cv2.threshold(img, iso_thresh, 1, cv2.THRESH_BINARY_INV)
    return feature_map


def _morph_filtering(img, kernel_size=5, kernel_shape="square") -> np.ndarray:
    """Filters thresholding noise"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = opening(img, kernel)
    closed = closing(opened, kernel)
    return closed


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
