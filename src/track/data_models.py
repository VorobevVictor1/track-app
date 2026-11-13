from dataclasses import dataclass, field
from numpy import empty
from numpy.typing import NDArray
from enum import StrEnum, Enum
from typing import NamedTuple
from napari.utils.transforms import Affine


# data needed for segmentation process
@dataclass(frozen=True)
class OutputTranforms2D:
    affine_matrix: NDArray[...] | Affine
    scale: tuple[float, float]
    translate: tuple[float, float]


@dataclass(frozen=True)
class SegmentationImageData:
    reflected_img: NDArray[...]
    transmitted_img: NDArray[...]
    output_layer_name: str
    transforms: OutputTranforms2D
    roi_mask: NDArray[...] | None = None


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
    labels: NDArray[...]


class ROIType(Enum):
    INCLUDE = 1
    EXCLUDE = -1


class ROI(NamedTuple):
    vertices: NDArray[...]
    shape_type: str
    roi_type: ROIType = ROIType.INCLUDE


@dataclass
class CalculationRequest:
    roi_mask: NDArray[...] | None
    px_per_um: float
    area_px: float
    tracks_map: NDArray[...] = field(default_factory=lambda: empty(0))


@dataclass(frozen=True)
class CalculationResult:
    num_of_tracks: int
    area_um: float
    tracks_density: float


# TODO: add integration with pymmcore_plus so objective types are not hardcoded
# objectives magnifications for my microscope
class Objective(StrEnum):
    X5 = "X5"
    X10 = "X10"
    X20 = "X20"
    X50 = "X50"
    X100 = "X100"


class SaveCalcData(NamedTuple):
    grain_number: str
    num_of_tracks_circ: int
    area_circ: float
    density_circ_um: float
    density_circ_cm: float
    num_of_tracks_poly: int
    area_poly: float
    density_poly_um: float
    density_poly_cm: float
