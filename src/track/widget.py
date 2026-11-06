from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from pathlib import Path
import re
from pymmcore_plus import CMMCorePlus
from useq import MDASequence, ZAboveBelow, Position

from magicgui.widgets import Container, create_widget, ComboBox
from napari.qt.threading import create_worker
from napari.utils.notifications import show_warning

from .data_models import (
    Objective,
    OutputTranforms2D,
    SegmentationImageData,
    SegmentationRequestClassic,
    SegmentationRequestNN,
    SegmentationResult,
    CalculationRequest,
    SaveCalcData,
)
from .services.segmentation import (
    SegmentationServiceClassic,
    SegmentationServiceNN,
)  # TODO: add segmentation mode selection
from .services.calculations import CalculationService
from .services.layers import NapariLayersService
from .services.roi_processing import shapes_to_mask_and_area, points_to_mask
from .services import app_configuration as app_conf
from .services.data_io import (
    uppend_result_table,
    save_source_data,
    save_screenshot,
)
from .services.focus_stacking import FocusStacker


# TODO: заменить на mmc.getPixelSizeUm в коде где надо. Добавить увеличения и объективы в mm_config
# lenses scales in px/um
OBJECTIVE_PX_PER_UM = {
    Objective.X5: 1.46,
    Objective.X10: 2.91,
    Objective.X20: 5.88,
    Objective.X50: 14.53,
    Objective.X100: 29.55,
}


def gen_grain_number(layer_name: str) -> str:
    # finds numbers in str
    match = re.search(r"\d+", layer_name)
    if match:
        grain_number = match.group(0)
    else:
        grain_number = "0"
    return grain_number


class MainWidget(Container):
    def __init__(self, viewer: "napari.Viewer", mmc_core: CMMCorePlus):
        super().__init__()

        self._viewer = viewer

        self._mmc = mmc_core
        try:
            self._mmc.mda.events.sequenceStarted.disconnect()
            self._mmc.mda.events.sequenceFinished.disconnect()
            self._mmc.mda.events.frameReady.disconnect()
        except TypeError:
            # Сигнал не был ни к чему подключен, игнорируем ошибку
            pass
        self._mmc.mda.events.frameReady.connect(self.on_mda_frame)
        self._mmc.mda.events.sequenceFinished.connect(self.on_mda_finished)

        self._frame_buffer = []
        self._pos_before_mda: Position = Position(
            x=self._mmc.getXPosition(),
            y=self._mmc.getYPosition(),
            z=self._mmc.getZPosition(),
        )

        self._segmentation_service = SegmentationServiceClassic()
        self._layers_service = NapariLayersService(self._viewer)
        self._calculations_service = CalculationService()
        self._focus_stacking_service = FocusStacker()

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
        self._align_button.clicked.connect(self._align_layers)
        self._manual_calc_button.clicked.connect(self._calc_manual)
        self._save_current_buton.clicked.connect(self._save_grain_data)
        self._load_widg.changed.connect(self._load_files)
        self._focus_stack.clicked.connect(self._stack_imgs)
        self._z_stack.clicked.connect(self._capture_z_stack)
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
                self._area_output,
                self._magnification_combo,
                self._swap_chb,
                self._align_button,
                self._z_stack,
                self._focus_stack,
                self._run_button,
                self._manual_calc_button,
                self._save_current_buton,
                self._load_widg,
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
        self._area_output = create_widget(
            label="ROI area in um^2",
            annotation=float,
            is_result=True,
            options={"value": 0},
        )
        self._magnification_combo = ComboBox(
            label="Objective",
            choices=[objective.value for objective in Objective],
            value=Objective.X100.value,
        )
        self._align_button = create_widget(label="Align", widget_type="PushButton")
        self._run_button = create_widget(
            label="Auto calculation", widget_type="PushButton"
        )
        self._manual_calc_button = create_widget(
            label="Calculate density", widget_type="PushButton"
        )
        self._save_current_buton = create_widget(
            label="Save grain data", widget_type="PushButton"
        )
        self._load_widg = create_widget(
            label="Load data",
            annotation=Path,
            options={"mode": "rm"},
            value=app_conf.config_instance.settings.sample_dir,
        )
        self._focus_stack = create_widget(label="Focus stack", widget_type="PushButton")
        self._z_stack = create_widget(label="z_stack", widget_type="PushButton")
        self._swap_chb = create_widget(label="Swap refl&trans", widget_type="CheckBox")

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

        if not self._swap_chb.value:
            right_layer = tr_layer
            right_img = tr_img
        else:
            right_layer = refl_layer
            right_img = refl_img

        # prepare ROI mask
        roi_shapes_layer = self._roi_layer.value
        mask, area = shapes_to_mask_and_area(
            roi_shapes_layer, right_img.shape[:2], right_layer.world_to_data
        )

        # prepare objective scale
        objective = Objective(self._magnification_combo.value)
        px_per_um = OBJECTIVE_PX_PER_UM[objective]

        image_transforms = OutputTranforms2D(
            affine_matrix=right_layer.affine,
            scale=right_layer.scale,
            translate=right_layer.translate,
        )
        grain_number = gen_grain_number(refl_layer.name)

        image_data = SegmentationImageData(
            reflected_img=refl_img,
            transmitted_img=tr_img,
            roi_mask=mask,
            output_layer_name=f"{grain_number}_segmented",
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
        self._area_output.value = calc_result.area_um

        self._current_calc_request = None

    def _align_layers(self):
        refl_image_layer = self._refl_image_layer.value
        tr_image_layer = self._transm_image_layer.value

        if refl_image_layer is None or tr_image_layer is None:
            show_warning("There is nothing to allign")
            return

        # if not checked then base mode: trans on the right, refl on the left
        if not self._swap_chb.value:
            left_layer = refl_image_layer
            right_layer = tr_image_layer
        else:
            left_layer = tr_image_layer
            right_layer = refl_image_layer

        try:
            translation = self._layers_service.calc_layer_translation(left_layer)
        except Exception as exc:
            show_warning(f"Translation calculation failed: {exc}")
            return

        other_layers = []
        try:  # this layer may be not created yet
            other_layers.append(self._viewer.layers[self._last_output_layer_name])
        except:
            pass

        self._layers_service.align_layers(
            left_layer=left_layer,
            right_layer=right_layer,
            other_layers=other_layers,
            translation=translation,
        )

    def _calc_manual(self):
        # get layers
        roi_layer = self._roi_layer.value
        track_points_layer = self._track_points_layer.value
        if track_points_layer is None:
            raise ValueError("Track points layer is not stated.")

        # get layers data
        track_points_list = track_points_layer.data
        if track_points_list.size == 0:
            self._density_output.value = 0
            self._tracks_num_output.value = 0
            self._area_output.value = 0
            return

        if not self._swap_chb.value:
            right_layer = self._transm_image_layer.value
        else:
            right_layer = self._refl_image_layer.value

        if right_layer is not None:
            shape = right_layer.data.shape[:2]
            target_world_to_data = right_layer.world_to_data
        else:
            raise ValueError("There is no tracks image.")
        # check whether point in roi
        roi_mask, area = shapes_to_mask_and_area(roi_layer, shape, target_world_to_data)

        points_mask = points_to_mask(track_points_layer, shape, target_world_to_data)

        grain_number = gen_grain_number(self._refl_image_layer.value.name)
        result_layer_name = f"{grain_number}_manual_calc"

        self._layers_service.upsert_labels(
            points_mask * roi_mask,
            result_layer_name,
            OutputTranforms2D(
                affine_matrix=right_layer.affine,
                scale=right_layer.scale,
                translate=right_layer.translate,
            ),
        )
        self._last_output_layer_name = result_layer_name

        # prepare objective scale
        objective = Objective(self._magnification_combo.value)
        px_per_um = OBJECTIVE_PX_PER_UM[objective]

        calc_result = self._calculations_service.process_calc_request(
            CalculationRequest(
                roi_mask=roi_mask,
                px_per_um=px_per_um,
                area_px=area,
                tracks_map=points_mask,
            )
        )

        self._density_output.value = calc_result.tracks_density
        self._tracks_num_output.value = calc_result.num_of_tracks
        self._area_output.value = calc_result.area_um

    def _save_grain_data(self):
        self._save_current_buton.enabled = False
        try:
            refl_im_layer = self._refl_image_layer.value
            trans_im_layer = self._transm_image_layer.value
            ldt_list = [
                refl_im_layer,
                trans_im_layer,
                self._roi_layer.value,
                self._track_points_layer.value,
            ]
            try:
                ldt_list.append(self._viewer.layers[self._last_output_layer_name])
            except:
                pass

            grain_number = gen_grain_number(refl_im_layer.name)

            dir_path = Path(self._config_service.settings.sample_dir).resolve()

            save_source_data(
                ldt_list, dir_path / "source_data" / ("GRAIN_" + grain_number)
            )
            data_to_save = SaveCalcData(
                grain_number,
                int(self._tracks_num_output.value),
                float(self._area_output.value),
                float(self._density_output.value),
                round(float(self._density_output.value) * 10**8, -3),
                int(self._tracks_num_output.value),
                float(self._area_output.value),
                float(self._density_output.value),
                round(float(self._density_output.value) * 10**8, -3),
            )
            uppend_result_table(
                data_to_save,
                dir_path / "подсчет.csv",
            )

            screenshot = self._viewer.export_figure(path=None, scale_factor=0.5)
            save_screenshot(
                screenshot, dir_path / "screenshots" / (grain_number + ".jpg")
            )
        finally:
            self._save_current_buton.enabled = True

    def _load_files(self, paths: tuple[Path, ...]):
        paths_set = set(paths)

        for path in paths:
            if path.suffix != ".toml":
                config_filename = f"{path.stem}_meta.toml"
                settings_path = path.with_name(config_filename)
                if settings_path in paths_set:
                    from tomllib import load

                    with open(settings_path, "rb") as f:
                        meta_dict = load(f)
                else:
                    meta_dict = {}
                try:
                    self._viewer.open(path, **meta_dict)
                except Exception as e:
                    show_warning(f"Error: {e}")

    def _stack_imgs(self):
        imgs_list = self._viewer.layers["stack"].data
        stack = self._focus_stacking_service.focus_stack(imgs_list)
        self._viewer.add_image(stack, name="stacked")

    def on_mda_frame(self, img):
        self._frame_buffer.append(img)

    def on_mda_finished(self):
        from numpy import asarray

        self._viewer.add_image(
            asarray(self._frame_buffer), multiscale=False, name="stack"
        )
        self._frame_buffer.clear()
        if self._pos_before_mda.x is not None and self._pos_before_mda.y is not None:
            self._mmc.setXYPosition(self._pos_before_mda.x, self._pos_before_mda.y)
        else:
            show_warning("Saved x or y position is None")
        if self._pos_before_mda.z is not None:
            self._mmc.setZPosition(self._pos_before_mda.z)
        else:
            show_warning("Saved z position is None")

    def _capture_z_stack(self):
        self._pos_before_mda = Position(
            x=self._mmc.getXPosition(),
            y=self._mmc.getYPosition(),
            z=self._mmc.getZPosition(),
        )
        mda_sequence = MDASequence(
            z_plan=ZAboveBelow(go_up=True, above=3.0, below=0.0, step=0.3),
            stage_positions=[self._pos_before_mda],
        )
        self._mmc.run_mda(mda_sequence)
