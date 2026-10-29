from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    import napari.layers
import numpy as np
from ..data_models import OutputTranforms2D


class NapariLayersService:
    def __init__(self, viewer: "napari.Viewer") -> None:
        self._viewer = viewer

    def calc_layer_translation(self, layer: "napari.layers.Image") -> np.ndarray:
        """Calculates new translation values for some layer based on given layer shape"""
        if layer is not None:
            current_translation = np.array(layer.translate)
            # check whether 2d and not rgb
            if layer.ndim == 2 and layer.rgb:
                current_translation[-1] = (
                    layer.data.shape[  # in napari axes order is inverted
                        -2  # last axis is 3 for 3 rgb channnels
                    ]  # translate layers for the width of camera image
                )
            elif layer.ndim == 2 and not layer.rgb:
                current_translation[-1] = layer.data.shape[-1]
            else:
                raise ValueError("Currently only 2D images supported")
            return current_translation

    def translate_layers(
        self,
        left_layer: "napari.layers.Layer",
        right_layer: "napari.layers.Layer",
        measurements_layers: list['napari.layers.Layer | None'],
        translation: np.ndarray
    ) -> None:
        left_layer.translate = np.zeros_like(translation)
        right_layer.translate = translation
        for layer in measurements_layers:
            if layer is not None:
                layer.translate = translation

    def upsert_labels(
        self, data: np.ndarray, name: str, output_transforms: OutputTranforms2D
    ) -> "napari.layers.Layer":
        try:
            layer = self._viewer.layers[name]
            layer.data = data
        except KeyError:
            layer = self._viewer.add_labels(data, name=name)

        layer.translate = output_transforms.translate
        layer.affine = output_transforms.affine_matrix
        layer.scale = output_transforms.scale
        return layer
