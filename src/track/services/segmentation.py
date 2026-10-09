from abc import ABC, abstractmethod
from ..data_models import (
    SegmentationRequestClassic,
    SegmentationRequestNN,
    SegmentationResult,
)
import numpy as np
import cv2
from track.modelling.onnx_inference import predict_mask
from skimage.filters import threshold_isodata
from skimage.measure import label
from skimage.morphology import closing, opening


class BaseSegmentationService(ABC):
    @abstractmethod
    def run(
        self, request: SegmentationRequestClassic | SegmentationRequestNN
    ) -> SegmentationResult:
        """This method must call segmentation implementation"""
        pass


# classic CV thresholding + postprocessing for instance segmentation
class SegmentationServiceClassic(BaseSegmentationService):
    def run(self, request: SegmentationRequestClassic) -> SegmentationResult:
        labels = instance_segment(
            img_reflected=request.segmentation_image_data.reflected_img,
            img_transmitted=request.segmentation_image_data.transmitted_img,
            mask=request.segmentation_image_data.roi_mask,
            sigma=request.sigma,
            gray_level=request.gray_level,
        )

        return SegmentationResult(labels)


# neural network for instance segmentation
class SegmentationServiceNN(BaseSegmentationService):
    def run(self, request: SegmentationRequestNN) -> SegmentationResult:
        labels = segm_with_nn(
            img_reflected=request.segmentation_image_data.reflected_img,
            img_transmitted=request.segmentation_image_data.transmitted_img,
            mask=request.segmentation_image_data.roi_mask,
            threshold=request.logit_thresh,
        )

        return SegmentationResult(labels)


def instance_segment(
    img_reflected: np.ndarray,
    img_transmitted: np.ndarray,
    mask: np.ndarray | None,
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
    mask: np.ndarray | None,
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
