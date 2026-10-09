from dataclasses import dataclass
from numpy import ndarray
from enum import Enum


# data needed for segmentation process
@dataclass(frozen=True)
class SegmentationImageData:
    reflected_img: ndarray
    transmitted_img: ndarray
    roi_mask: ndarray | None = None
    # output_layer_name: str
    # translation: tuple[float, ...]
    # area_px: float | None = 0
    # px_per_um: float


@dataclass(frozen=True)
class SegmentationRequestClassic:
    segmentation_image_data: SegmentationImageData
    sigma: float = 25.0
    gray_level: int = 128


@dataclass(frozen=True)
class SegmentationRequestNN:
    segmentation_image_data: SegmentationImageData
    logit_thresh: float = 0.0


# result of segmentation process
@dataclass(frozen=True)
class SegmentationResult:
    labels: ndarray
    # density: float = 0


# TODO: add integration with pymmcore_plus so objective types are not hardcoded
# objectives magnifications for my microscope
class Objective(str, Enum):
    X5 = "X5"
    X10 = "X10"
    X20 = "X20"
    X50 = "X50"
    X100 = "X100"
