from dataclasses import dataclass, field
from numpy import ndarray, empty
from enum import Enum
from typing import NamedTuple


# data needed for segmentation process
@dataclass(frozen=True)
class SegmentationImageData:
    reflected_img: ndarray
    transmitted_img: ndarray
    output_layer_name: str
    translation: ndarray
    roi_mask: ndarray | None = None


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


class ROIType(Enum):
    INCLUDE = 1
    EXCLUDE = -1


class ROI(NamedTuple):
    vertices: ndarray
    shape_type: str
    roi_type: ROIType = ROIType.INCLUDE


@dataclass
class CalculationRequest:
    roi_mask: ndarray | None
    px_per_um: float
    area_px: float
    tracks_map: ndarray = field(default_factory=lambda: empty(0))


@dataclass(frozen=True)
class CalculationResult:
    num_of_tracks: int
    tracks_density: float


# TODO: add integration with pymmcore_plus so objective types are not hardcoded
# objectives magnifications for my microscope
class Objective(str, Enum):
    X5 = "X5"
    X10 = "X10"
    X20 = "X20"
    X50 = "X50"
    X100 = "X100"
