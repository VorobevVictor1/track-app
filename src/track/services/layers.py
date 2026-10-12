from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari
    import napari.layers
import numpy as np


class NapariLayersService:
    def __init__(self, viewer: 'napari.Viewer') -> None:
        self._viewer = viewer

    def calc_layer_translation(self, layer: 'napari.layers.Image') -> np.ndarray:
        """Calculates new translation values for some layer based on given layer shape"""
        if layer is not None:
            current_translation = np.array(layer.translate, copy = True)
            #if 2d not rgb, then last axis
            if layer.data.ndim == 2:
                current_translation[-1] = layer.data.shape[
                    -1
                ]  # Modifying the last dimension (X)
            else:
                #if rgb then second axis  TODO: fix for 3d rgb probably should check rgb property
                current_translation[-1] = layer.data.shape[
                    1
                ]  # Modifying the last dimension (X)
            return current_translation
    
    def translate_layers(self, layers_list: list['napari.layers.Layer'], translation: np.ndarray) -> None:
        for layer in layers_list:
            if layer is not None:
                layer.translate = translation

    def upsert_labels(self, data: np.ndarray, name: str, translate) -> 'napari.layers.Layer':
        try:
            layer = self._viewer.layers[name]
            layer.data = data
        except KeyError:
            layer = self._viewer.add_labels(data, name=name)

        layer.translate = translate
        return layer