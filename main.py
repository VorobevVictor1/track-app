import napari
from magicgui.widgets import Container, create_widget
from napari.qt.threading import create_worker
from src.track.funcs import *


class AutoCountWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._transm_image_layer = create_widget(
            label="Transmitted", annotation="napari.layers.Image"
        )
        self._refl_image_layer = create_widget(
            label="Reflected", annotation="napari.layers.Image"
        )
        self._roi_layer = create_widget(
            label="ROI (Shapes)", annotation="napari.layers.Shapes"
        )
        self._sigma_slider = create_widget(
            label="Sigma",
            annotation=float,
            options={
                "value": 25.0,
                "min": -50.0,
            },
        )
        self._gray_slider = create_widget(
            label="Gray level", annotation=int, options={"value": 128}
        )
        self._density_output = create_widget(
            label="Tracks density num/px^2",
            annotation=float,
            is_result=True,
            options={"value": 0.0},
        )
        self._run_button = create_widget(label="Run", widget_type="PushButton")
        # connect your own callbacks
        self._run_button.changed.connect(self._process_im)
        # append into/extend the container with your widgets
        self.extend(
            [
                self._transm_image_layer,
                self._refl_image_layer,
                self._roi_layer,
                self._sigma_slider,
                self._gray_slider,
                self._density_output,
                self._run_button,
            ]
        )

    def _upd_widget(self, img):
        name = self._transm_image_layer.value.name + "_segmented"
        # Update existing layer (if present) or add new labels layer
        if name in self._viewer.layers:
            self._viewer.layers[name].data = img
        else:
            self._viewer.add_labels(img, name=name)

        shapes_layer = self._roi_layer.value
        area = None
        if shapes_layer is not None:
            if shapes_layer.nshapes == 1:
                area = measure_area(shapes_layer.data, shapes_layer.shape_type)
            else:
                raise Exception
        self._density_output.value = segm_postprocessing(img, area)

    def _process_im(self):
        tr_image_layer = self._transm_image_layer.value
        refl_image_layer = self._refl_image_layer.value
        if refl_image_layer is None or tr_image_layer is None:
            return
        sigma = self._sigma_slider.value
        gray = self._gray_slider.value
        img_tr = tr_image_layer.data
        img_re = refl_image_layer.data

        shapes_layer = self._roi_layer.value
        mask = None
        if shapes_layer is not None:
            if shapes_layer.nshapes == 0:
                pass
            elif shapes_layer.nshapes == 1:
                mask = shapes_layer.to_masks(mask_shape=img_tr.shape[:2])
                mask = np.reshape(mask, img_tr.shape[:2])
            else:
                raise Exception

        worker = create_worker(
            instance_segment, img_tr, img_re, mask, sigma, gray, _start_thread=False
        )
        # worker = create_worker(segm_with_nn, img_tr, img_re, mask, sigma, _start_thread=False,)
        worker.returned.connect(self._upd_widget)
        worker.start()


if __name__ == "__main__":
    # Create a `viewer`
    viewer = napari.Viewer()
    # Instantiate your widget
    my_widg = AutoCountWidget(viewer)
    # Add widget to `viewer`
    viewer.window.add_dock_widget(my_widg)
    napari.run()
