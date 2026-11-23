from ..data_model.grain_cls import Grain
from dataclasses import dataclass, field
from enum import StrEnum
from napari.layers import Layer, Image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari import Viewer


class LayerRole(StrEnum):
    REFL_SURFACE = "refl_surface_image"


@dataclass
class GrainBinding:
    grain: Grain
    layers_dict: dict[LayerRole, Layer] = field(default_factory=dict)


class NapariGrainConnector:
    def __init__(self, viewer: "Viewer") -> None:
        self._viewer = viewer
        self._bindings_dict: dict[int, GrainBinding] = {}

    def add_grain(self, grain: Grain):
        if grain.number in self._bindings_dict:
            raise ValueError(f"Grain {grain.number} already exists.")
        l = self._viewer.add_image(
            data=grain.refl_surface_image, rgb=True, units="micrometer"
        )
        if isinstance(l, Image):
            l.editable = False
            binding = GrainBinding(grain, {LayerRole.REFL_SURFACE: l})
            self._bindings_dict[grain.number] = binding

    def get_grain_binding(self, number: int):
        if number in self._bindings_dict:
            return self._bindings_dict[number]
        else:
            return None
