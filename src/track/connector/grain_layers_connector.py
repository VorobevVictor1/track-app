from ..data_model.grain_cls import Grain
from ..data_model.data_models import ROI, ROIType, ROIShapeTypes
from dataclasses import dataclass, field
from enum import StrEnum
import numpy.typing as npt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari import Viewer
    from napari.layers import Layer


class LayerRole(StrEnum):
    REFL_SURFACE_IMAGE = "refl_surface_image"
    TRANSM_SURFACE_IMAGE = "transm_surface_image"
    TRACK_PITS_LABELS = "track_pits_mask"
    ROI_SHAPES_INTERACTIVE = "roi_list"
    ROI_SHAPES_GHOSTING = "roi_list ghosting"
    TRACK_POINTS_INTERACTIVE = "track_points_array"
    TRACK_POINTS_GHOSTING = "track_points_array ghosting"
    TRANSM_STACK_IMAGES = "transm_stack"
    REFL_EDF_IMAGE = "refl_edf"
    TRANSM_EDF_IMAGE = "transm_edf"


@dataclass
class GrainBinding:
    grain: Grain
    layers_dict: dict[LayerRole, "Layer"] = field(default_factory=dict)

def convert_shapes_to_ROI(shape_vertices: list[npt.NDArray[np.float32]], shape_types: list[str], roi_types: list[ROIType]):
    roi_list = []

    for shape, sh_type, r_type in zip(shape_vertices, shape_types, roi_types):
        if sh_type in ROIShapeTypes:
            roi_list.append(
                ROI(
                    vertices=shape,
                    shape_type=sh_type,
                    roi_type=r_type
                )
            )
    return roi_list

class NapariGrainConnector:
    def __init__(self, viewer: "Viewer") -> None:
        self._viewer = viewer
        self._bindings_dict: dict[int, GrainBinding] = {}
        self.active_grain_binding: GrainBinding | None = None

    def create_grain(self, grain: Grain, scale: float, units: str):
        if grain.number in self._bindings_dict:
            raise ValueError(f"Grain {grain.number} already exists.")
        binding = GrainBinding(grain)
        
        binding = self._add_images(binding, scale, units)
        binding = self._add_points(binding, scale, units)
        binding = self._add_shapes(binding, scale, units)

        self._bindings_dict[grain.number] = binding

    def _add_images(self, binding: GrainBinding, scale: float, units: str) -> GrainBinding:
        for role in (LayerRole.TRANSM_SURFACE_IMAGE, LayerRole.REFL_SURFACE_IMAGE):
            layer = self._viewer.add_image(
                data=getattr(binding.grain, role), rgb=True, units=units, name=f"{binding.grain.number} {role}",
                scale=(scale, scale)
            )
            #should always be true because we add only 1 image
            if not isinstance(layer, list):
                layer.editable = False
                binding.layers_dict[role] = layer
        
        return binding
    
    def _add_points(self, binding: GrainBinding, scale: float, units: str) -> GrainBinding:
        for role in (LayerRole.TRACK_POINTS_INTERACTIVE, LayerRole.TRACK_POINTS_GHOSTING):
            layer = self._viewer.add_points(
                data=getattr(binding.grain, role.split(' ')[0]), units=units, name=f"{binding.grain.number} {role}",
                scale=(scale, scale),
            )
            binding.layers_dict[role] = layer
            if role == LayerRole.TRACK_POINTS_GHOSTING:
                layer.editable = False
        
        def _on_points(event):
            binding.grain.track_points_array = event.source.data
            binding.layers_dict[LayerRole.TRACK_POINTS_GHOSTING].data = event.source.data
        
        binding.layers_dict[LayerRole.TRACK_POINTS_INTERACTIVE].events.data.connect(_on_points)
        return binding

    def _add_shapes(self, binding: GrainBinding, scale: float, units: str) -> GrainBinding:
        for role in (LayerRole.ROI_SHAPES_INTERACTIVE, LayerRole.ROI_SHAPES_GHOSTING):
            layer = self._viewer.add_shapes(
                data=getattr(binding.grain, role.split(' ')[0]), units=units, name=f"{binding.grain.number} {role}",
                scale=(scale, scale),
            )
            binding.layers_dict[role] = layer
            if role == LayerRole.ROI_SHAPES_GHOSTING:
                layer.editable = False
        def _on_shapes(event):
            r_types = [ROIType.INCLUDE] * event.source.nshapes
            binding.grain.roi_list = convert_shapes_to_ROI(event.source.data, event.source.shape_type, r_types)
            binding.layers_dict[LayerRole.ROI_SHAPES_GHOSTING].remove(list(range(binding.layers_dict[LayerRole.ROI_SHAPES_GHOSTING].nshapes)))
            binding.layers_dict[LayerRole.ROI_SHAPES_GHOSTING].add(event.source.data, shape_type=event.source.shape_type)
        binding.layers_dict[LayerRole.ROI_SHAPES_INTERACTIVE].events.data.connect(_on_shapes)
        return binding
    
    def get_grain_binding(self, number: int):
        if number in self._bindings_dict:
            return self._bindings_dict[number]
        else:
            return None

    def set_active_grain(self, number: int):
        self.active_grain_binding = self.get_grain_binding(number) 