import napari
import numpy as np

viewer = napari.Viewer()
layer = viewer.add_image(np.random.random((512, 512)))


@layer.mouse_drag_callbacks.append
def update_layer(layer, event):
    layer.data = np.random.random((512, 512))


napari.run()
