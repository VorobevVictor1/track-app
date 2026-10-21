from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from pathlib import Path

from magicgui.widgets import Container, create_widget, ComboBox
from napari.qt.threading import create_worker
from napari.utils.notifications import show_warning
from napari.experimental import link_layers

from .data_models import (
    Objective,
    OutputTranforms2D,
    SegmentationImageData,
    SegmentationRequestClassic,
    SegmentationRequestNN,
    SegmentationResult,
    CalculationRequest,
)
from .services.segmentation import (
    SegmentationServiceClassic,
    SegmentationServiceNN,
)  # TODO: add segmentation mode selection
from .services.calculations import CalculationService
from .services.layers import NapariLayersService
from .services.roi_processing import shapes_to_mask_and_area, points_to_mask
from .services import app_configuration as app_conf
from .services.grain_saver import save_img



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
        self._layers_service = NapariLayersService(self._viewer)
        self._calculations_service = CalculationService()

        if app_conf.config_instance is not None:
            self._config_service = app_conf.config_instance
        else:
            raise ValueError("Config object is not initialized")

        self._worker = None
        self._current_segm_request: (
            SegmentationRequestClassic | SegmentationRequestNN | None
        ) = None
        self._current_calc_request: CalculationRequest | None = None
        self._last_output_layer_name: str | None = None
        self._build_ui()

        # connect your own callbacks
        self._run_button.clicked.connect(self._start_segmentation)
        self._translate_button.clicked.connect(self._align_layers)
        self._manual_calc_button.clicked.connect(self._calc_manual)
        self._save_current_buton.clicked.connect(self._save_grain_data)
        # append into/extend the container with your widgets
        self.extend(
            [
                self._refl_image_layer,
                self._transm_image_layer,
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
                self._save_current_buton,
            ]
        )

    def _build_ui(self):
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
            annotation=float,
            is_result=True,
            options={"value": 0.0},
        )
        self._tracks_num_output = create_widget(
            label="Amount of tracks",
            annotation=int,
            is_result=True,
            options={"value": 0},
        )
        self._magnification_combo = ComboBox(
            label="Objective",
            choices=[objective.value for objective in Objective],
            value=Objective.X100.value,
        )
        self._translate_button = create_widget(
            label="Translate", widget_type="PushButton"
        )
        self._run_button = create_widget(
            label="Auto calculation", widget_type="PushButton"
        )
        self._manual_calc_button = create_widget(
            label="Calculate density", widget_type="PushButton"
        )
        self._save_current_buton = create_widget(
            label="Save grain data", widget_type="PushButton"
        )

    def _start_segmentation(self):
        """Starts segmentation worker process"""
        if self._worker is not None and self._worker.is_running:
            return

        try:
            segm_request, calc_request = self._build_requests()
        except ValueError as exc:
            show_warning(str(exc))
            return

        self._current_segm_request = segm_request
        self._current_calc_request = calc_request
        self._run_button.enabled = False

        worker = create_worker(
            self._segmentation_service.run, segm_request, _start_thread=False
        )
        worker.returned.connect(self._on_segmentation_result)
        worker.errored.connect(self._on_segmentation_error)
        worker.finished.connect(self._on_worker_finished)

        self._worker = worker
        worker.start()

    def _build_requests(
        self,
    ) -> tuple[SegmentationRequestClassic | SegmentationRequestNN, CalculationRequest]:
        """Creates a data snapshot for segmentation worker"""
        # prepare images
        refl_layer = self._refl_image_layer.value
        tr_layer = self._transm_image_layer.value

        if tr_layer is None or refl_layer is None:
            raise ValueError("Select both transmitted and reflected image")

        refl_img = refl_layer.data
        tr_img = tr_layer.data

        if refl_layer.ndim > 2 or tr_layer.ndim > 2:
            raise ValueError("Only 2D images currently supported")

        if refl_img.shape != tr_img.shape:
            raise ValueError("Images must have the same shape")

        # prepare ROI mask
        roi_shapes_layer = self._roi_layer.value
        mask, area = shapes_to_mask_and_area(
            roi_shapes_layer, tr_img.shape[:2], refl_layer.world_to_data
        )

        # prepare objective scale
        objective = Objective(self._magnification_combo.value)
        px_per_um = OBJECTIVE_PX_PER_UM[objective]

        image_transforms = OutputTranforms2D(
            affine_matrix=refl_layer.affine,
            scale=refl_layer.scale,
            translate=refl_layer.translate,
        )
        image_data = SegmentationImageData(
            reflected_img=refl_img,
            transmitted_img=tr_img,
            roi_mask=mask,
            output_layer_name=f"{refl_layer.name}_segmented",
            transforms=image_transforms,
        )
        segm_request = SegmentationRequestClassic(
            segmentation_image_data=image_data,
            sigma=self._sigma_slider.value,
            gray_level=self._gray_slider.value,
        )
        calc_request = CalculationRequest(
            roi_mask=mask, px_per_um=px_per_um, area_px=area
        )
        return segm_request, calc_request

    def _on_segmentation_error(self, exc):
        show_warning(f"Segmentation failed: {exc}")

    def _on_worker_finished(self, status=None):
        self._worker = None
        self._current_segm_request = None
        self._run_button.enabled = True

    def _on_segmentation_result(self, result: SegmentationResult):
        # add segmentation result to calculation request
        self._current_calc_request.tracks_map = result.labels

        segm_request = self._current_segm_request

        name = segm_request.segmentation_image_data.output_layer_name
        output_tranform = segm_request.segmentation_image_data.transforms
        # Update existing layer (if present) or add new labels layer
        self._layers_service.upsert_labels(
            result.labels,
            name,
            output_tranform,
        )
        self._last_output_layer_name = name

        calc_result = self._calculations_service.process_calc_request(
            self._current_calc_request
        )

        self._tracks_num_output.value = calc_result.num_of_tracks
        self._density_output.value = calc_result.tracks_density

        self._current_calc_request = None

    def _align_layers(self):
        refl_image_layer = self._refl_image_layer.value

        if refl_image_layer is None:
            show_warning("There is nothing to allign")
            return

        try:
            translation = self._layers_service.calc_layer_translation(refl_image_layer)
        except Exception as exc:
            show_warning(f"Translation calculation failed: {exc}")
            return

        tr_image_layer = self._transm_image_layer.value
        roi_layer = self._roi_layer.value
        points_layer = self._track_points_layer.value

        layers_list = [tr_image_layer, refl_image_layer, roi_layer, points_layer]
        if (
            self._last_output_layer_name is not None
            and self._last_output_layer_name in self._viewer.layers
        ):
            layers_list.append(self._viewer.layers[self._last_output_layer_name])

        self._layers_service.translate_layers(layers_list, translation)

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
        roi_shapes_list = roi_layer.data
        if len(roi_shapes_list) == 0:
            self._density_output.value = 0
        if self._refl_image_layer.value is not None:
            shape = self._refl_image_layer.value.data.shape[:2]
            target_world_to_data = self._refl_image_layer.value.world_to_data
        else:
            raise ValueError("There is no tracks image.")
        # check whether point in roi
        roi_mask, area = shapes_to_mask_and_area(roi_layer, shape, target_world_to_data)

        points_mask = points_to_mask(track_points_layer, shape, target_world_to_data)

        self._layers_service.upsert_labels(
            points_mask * roi_mask,
            "manual_calc",
            OutputTranforms2D(
                affine_matrix=self._refl_image_layer.value.affine,  # returnes identical transform
                scale=self._refl_image_layer.value.scale,
                translate=self._refl_image_layer.value.translate,
            ),
        )
        self._last_output_layer_name = "manual_calc"

        # prepare objective scale
        objective = Objective(self._magnification_combo.value)
        px_per_um = OBJECTIVE_PX_PER_UM[objective]

        response = self._calculations_service.process_calc_request(
            CalculationRequest(
                roi_mask=roi_mask,
                px_per_um=px_per_um,
                area_px=area,
                tracks_map=points_mask,
            )
        )

        self._density_output.value = response.tracks_density
        self._tracks_num_output.value = response.num_of_tracks

    def _save_grain_data(self):
        dir_path = Path(self._config_service.settings.sample_dir).resolve()
        img_extension = '.png'

        refl_data = self._refl_image_layer.value.data
        refl_name = self._refl_image_layer.value.name + img_extension
        refl_path = dir_path / refl_name

        transm_data = self._transm_image_layer.value.data
        transm_name = self._transm_image_layer.value.name + img_extension
        transm_path = dir_path / transm_name
        
        save_img(refl_data, refl_path, img_extension)
        save_img(transm_data, transm_path, img_extension)
        