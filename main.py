import napari
from magicgui.widgets import Container, create_widget, ComboBox
from napari.qt.threading import create_worker, FunctionWorker
from napari.utils.notifications import show_warning
from src.track.funcs import (
    density_calc,
    measure_area,
    translate_layer,
    frame_postproc,
)
from src.track.data_models import (
    Objective,
    SegmentationImageData,
    SegmentationRequestClassic,
    SegmentationRequestNN,
    SegmentationResult,
)
from src.track.services.segmentation import (
    SegmentationServiceClassic,
    SegmentationServiceNN,
)
from superqt.utils import ensure_main_thread
import numpy as np

from napari.experimental import link_layers


# TODO: заменить на mmc.getPixelSizeUm в коде где надо. Добавить увеличения и объективы в mm_config
# lenses scales in px/um
OBJECTIVE_PX_PER_UM = {
    Objective.X5: 1.46,
    Objective.X10: 2.91,
    Objective.X20: 5.88,
    Objective.X50: 14.53,
    Objective.X100: 29.55,
}


class MainWidget(Container):
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__()
        
        self._viewer = viewer
        self._segmentation_service = SegmentationServiceClassic()
        self._worker = None
        self._current_request: SegmentationRequestClassic|SegmentationRequestNN|None = None

        self._buiild_ui()

        # connect your own callbacks
        self._run_button.clicked.connect(self._start_segmentation)
        self._translate_button.clicked.connect(self._translate_photos)
        self._manual_calc_button.clicked.connect(self._calc_manual)
        # append into/extend the container with your widgets
        self.extend(
            [
                self._transm_image_layer,
                self._refl_image_layer,
                self._roi_layer,
                self._track_points_layer,
                self._sigma_slider,
                self._gray_slider,
                self._density_output,
                self._tracks_num_output,
                self._magnification_combo,
                self._translate_button,
                self._run_button,
                self._manual_calc_button,
            ]
        )

    def _buiild_ui(self):
        """Initializes ui elements for widget"""
        self._transm_image_layer = create_widget(
            label="Transmitted", annotation="napari.layers.Image"
        )
        self._refl_image_layer = create_widget(
            label="Reflected", annotation="napari.layers.Image"
        )
        self._roi_layer = create_widget(
            label="ROI (Shapes)", annotation="napari.layers.Shapes"
        )
        self._track_points_layer = create_widget(
            label="Track points layer", annotation="napari.layers.Points"
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
            annotation=list[np.float32],
            is_result=True,
            options={"value": 0.0},
        )
        self._tracks_num_output = create_widget(
            label="Amount of tracks",
            annotation=list[np.uint],
            is_result=True,
            options={"value": 0},
        )
        self._run_button = create_widget(label="Run", widget_type="PushButton")
        self._magnification_combo = ComboBox(
            label="Objective",
            choices=[objective.value for objective in Objective],
            value=Objective.X100.value,
        )
        self._translate_button = create_widget(
            label="Translate", widget_type="PushButton"
        )
        self._manual_calc_button = create_widget(
            label="Calculate density", widget_type="PushButton"
        )

    def _start_segmentation(self):
        """Starts segmentation worker process"""
        if self._worker is not None and self._worker.is_running:
            return
        
        try:
            request = self._build_segmentation_request()
        except ValueError as exc:
            show_warning(str(exc))
            return
        
        self._current_request = request
        self._run_button.enabled = False

        worker = create_worker(
            self._segmentation_service.run,
            request,
            _start_thread=False
        )
        # worker.returned.connect(self._on_segmentation_result) TODO: implement
        worker.returned.connect(self._upd_widget)
        worker.errored.connect(self._on_segmentation_error)
        worker.finished.connect(self._on_worker_finished)

        self._worker = worker
        worker.start()

    def _build_segmentation_request(self) -> SegmentationRequestClassic|SegmentationRequestNN:
        """Creates a data snapshot for segmentation worker"""
        #prepare images
        refl_layer = self._refl_image_layer.value
        tr_layer = self._transm_image_layer.value

        if tr_layer is None or refl_layer is None:
            raise ValueError("Select both transmitted and reflected image")
        
        refl_img = refl_layer.data
        tr_img = tr_layer.data

        #TODO: rewrite so rgb images could be detected correctly
        if refl_img.ndim > 3 or tr_img.ndim > 3:
            raise ValueError("Only 2D images currently supported")
        
        if refl_img.shape != tr_img.shape:
            raise ValueError("Images must have the same shape")
        
        #prepare ROI mask
        roi_shapes_layer = self._roi_layer.value
        if roi_shapes_layer is None or roi_shapes_layer.nshapes == 0:
            mask = None
        elif roi_shapes_layer.nshapes > 1:
            raise ValueError("Only 1 ROI is currently supported") #TODO: implement several ROIs and excluding ROI
        else:
            #there is only 1 mask in masks ndarray so we get it by id
            mask = roi_shapes_layer.to_masks(mask_shape=tr_img.shape[:2])[0]

        image_data = SegmentationImageData(
            reflected_img=refl_img,
            transmitted_img=tr_img,
            roi_mask=mask
        )
        return SegmentationRequestClassic(
            segmentation_image_data=image_data,
            sigma=self._sigma_slider.value,
            gray_level=self._gray_slider.value
        )

    def  _on_segmentation_result(self, result: SegmentationResult):
        pass

    def _on_segmentation_error(self, exc):
        show_warning(f"Segmentation failed: {exc}")

    def _on_worker_finished(self):
        self._worker = None
        self._current_request = None
        self._run_button.enabled = True

    def _process_im(self):
        tr_image_layer = self._transm_image_layer.value
        refl_image_layer = self._refl_image_layer.value
        if refl_image_layer is None or tr_image_layer is None:
            raise ValueError
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
                raise Exception("Currently onlu one ROI is implemented")

        worker = create_worker(
            instance_segment, img_tr, img_re, mask, sigma, gray, _start_thread=False
        )
        # worker = create_worker(segm_with_nn, img_tr, img_re, mask, sigma, _start_thread=False,)
        worker.returned.connect(self._upd_widget)
        worker.start()

    def _upd_widget(self, result: SegmentationResult):
        img = result.labels
        name = self._transm_image_layer.value.name + "_segmented"
        # Update existing layer (if present) or add new labels layer
        translation = self._transm_image_layer.value.translate
        if name in self._viewer.layers:
            self._viewer.layers[name].data = img
            self._viewer.layers[name].translate = translation
        else:
            self._viewer.add_labels(img, name=name, translate=translation)

        shapes_layer = self._roi_layer.value
        if shapes_layer is not None:
            if shapes_layer.nshapes == 1:
                area = measure_area(shapes_layer.data, shapes_layer.shape_type)
            else:
                raise Exception
        objective = Objective(self._magnification_combo.value)
        px_per_um = OBJECTIVE_PX_PER_UM[objective]
        num_of_tracks = np.count_nonzero(np.unique(img))
        self._tracks_num_output.value = [num_of_tracks]
        self._density_output.value = [density_calc(num_of_tracks, area, px_per_um)]

    def _translate_photos(self):

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

    def _calc_manual(self):
        # get layers
        roi_layer = self._roi_layer.value
        track_points_layer = self._track_points_layer.value
        if roi_layer is None or track_points_layer is None:
            raise ValueError("ROI or Track layer are not stated.")

        # get layers data
        track_points_list = track_points_layer.data
        if track_points_list.size == 0:
            self._density_output.value = 0
            return
        if (track_points_list < 0).any():
            raise ValueError(
                "One of the tracks points has negative coordinate in layer coordinate system."
            )
        roi_shapes_list = roi_layer.data
        if len(roi_shapes_list) == 0:
            self._density_output.value = 0
        for shape in roi_shapes_list:
            if (shape < 0).any():
                raise ValueError(
                    "One of the ROI vertesies has negative coordinate in layer coordinate system."
                )

        # check whether point in roi
        roi_masks = roi_layer.to_masks()
        roi_shapes_types = roi_layer.shape_type
        densities_list = []
        tracks_nums_list = []
        for shape, mask, sh_type in zip(roi_shapes_list, roi_masks, roi_shapes_types):
            point_list = []
            for point in track_points_list:
                point_world = track_points_layer.data_to_world(point)
                point_shapes = roi_layer.world_to_data(point_world)
                try:
                    ids = point_shapes.round().astype(np.uint, casting="same_value")
                    if mask[ids[-2], ids[-1]]:
                        point_list.append(point_shapes)
                except:
                    pass
            area = measure_area([shape], [sh_type])
            objective = Objective(self._magnification_combo.value)
            px_per_um = OBJECTIVE_PX_PER_UM[objective]
            num_of_tracks = len(point_list)
            densities_list.append(density_calc(num_of_tracks, area, px_per_um))
            tracks_nums_list.append(num_of_tracks)

        self._density_output.value = densities_list
        self._tracks_num_output.value = tracks_nums_list
        # TODO: area recalculation based on affine transform


# Added custom frame postprocessing function
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
    elif len(data.shape) == 2:
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
    my_widg = MainWidget(viewer)

    # Add widget to `viewer`
    viewer.window.add_dock_widget(my_widg)
    napari.run()
