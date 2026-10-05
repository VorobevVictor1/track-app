import napari
from magicgui.widgets import Container, create_widget, ComboBox
from napari.qt.threading import create_worker
from src.track.funcs import *
from superqt.utils import ensure_main_thread


# lenses scales in px/um
LENS_SCALES = {"5X": 1.46, "10X": 2.91, "20X": 5.88, "50X": 14.53, "100X": 29.55}


class AutoCountWidget(Container):
    def __init__(self, viewer: "napari.Viewer"):
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
            label="Tracks density num/um^2",
            annotation=float,
            is_result=True,
            options={"value": 0.0},
        )
        self._run_button = create_widget(label="Run", widget_type="PushButton")
        self._magnification_combo = ComboBox(
            value=list(LENS_SCALES.keys())[-1],
            choices=LENS_SCALES.keys(),
            label="Objective",
        )
        self._translate_button = create_widget(
            label="Translate", widget_type="PushButton"
        )
        # connect your own callbacks
        self._run_button.clicked.connect(self._process_im)
        self._translate_button.clicked.connect(self._translate_photos)
        # append into/extend the container with your widgets
        self.extend(
            [
                self._transm_image_layer,
                self._refl_image_layer,
                self._roi_layer,
                self._sigma_slider,
                self._gray_slider,
                self._density_output,
                self._magnification_combo,
                self._translate_button,
                self._run_button,
            ]
        )

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

    def _upd_widget(self, img):
        name = self._transm_image_layer.value.name + "_segmented"
        # Update existing layer (if present) or add new labels layer
        translation = self._transm_image_layer.value.translate
        if name in self._viewer.layers:
            self._viewer.layers[name].data = img
            self._viewer.layers[name].translate = translation
        else:
            self._viewer.add_labels(img, name=name, translate=translation)

        shapes_layer = self._roi_layer.value
        area = None
        if shapes_layer is not None:
            if shapes_layer.nshapes == 1:
                area = measure_area(shapes_layer.data, shapes_layer.shape_type)
            else:
                raise Exception
        scale = LENS_SCALES.get(self._magnification_combo.value)
        self._density_output.value = segm_postprocessing(img, area, scale)

    def _translate_photos(self):
        # TODO: fix the alignment. now after every press it is translated to right. This is wrong behaviour. Translation should be done only once
        tr_image_layer = self._transm_image_layer.value
        translation = translate_layer(tr_image_layer)
        tr_image_layer.translate = translation

        refl_image_layer = self._refl_image_layer.value
        refl_image_layer.translate = translation

        roi_layer = self._roi_layer.value
        if roi_layer is not None:
            roi_layer.translate = translation

        name = tr_image_layer.name + "_segmented"
        # Update existing layer (if present) or add new labels layer
        if name in self._viewer.layers:
            self._viewer.layers[name].translate = translation


@ensure_main_thread  # type: ignore [untyped-decorator]
def _update_viewer(self, data: np.ndarray | None = None) -> None:
    """Update viewer with the latest image from the circular buffer."""
    if data is None:
        if self._mmc.getRemainingImageCount() == 0:
            return
        try:
            data = self._mmc.getLastImage()
        except (RuntimeError, IndexError):
            # circular buffer empty
            return
    if len(data.shape) == 2:
        data = frame_postproc(data)
    try:
        preview_layer = self.viewer.layers["preview"]
        preview_layer.data = data
    except KeyError:
        preview_layer = self.viewer.add_image(data, name="preview")

    preview_layer.metadata["mode"] = "preview"

    if (pix_size := self._mmc.getPixelSizeUm()) != 0:
        preview_layer.scale = (pix_size, pix_size)
    else:
        # return to default
        preview_layer.scale = [1.0, 1.0]

    if self._live_timer_id is None:
        self.viewer.reset_view()


if __name__ == "__main__":
    try:
        import napari_micromanager._core_link as core_link

        core_link.CoreViewerLink._update_viewer = _update_viewer
    except Exception as e:
        print(f"[ERROR] Не удалось применить патч: {e}")

    # Create a `viewer`
    viewer = napari.Viewer()
    # Instantiate your widget
    my_widg = AutoCountWidget(viewer)

    # Add widget to `viewer`
    viewer.window.add_dock_widget(my_widg)
    napari.run()
