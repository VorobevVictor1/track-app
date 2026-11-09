from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import napari
from qtpy.QtCore import QObject, Qt, QTimerEvent
from superqt.utils import ensure_main_thread

from napari_micromanager._mda_handler import _NapariMDAHandler


if TYPE_CHECKING:
    from collections.abc import Callable

    import napari.viewer
    import numpy as np
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.core.events._protocol import PSignalInstance

import numpy as np
from superqt.utils import ensure_main_thread


def frame_postproc(video_buffer: np.ndarray) -> np.ndarray:
    """
    Postprocessing of livestream frame from pymmcore-plus
    """
    height, width = video_buffer.shape
    if video_buffer.dtype == np.uint32:
        pix_type = np.uint8
    elif video_buffer.dtype == np.uint64:
        pix_type = np.uint16
    else:
        raise ValueError(
            f"Неизвестный тип пикселей для toupcam: {video_buffer.dtype!r}. "
            f"Допустимые: np.uint32, np.uint64."
        )

    # 1. Раскладываем матрицу uint32 на 4 отдельных байта по третьей оси.
    # Теперь массив имеет форму [Height, Width, 4] и тип uint8.
    bytes_array = video_buffer.view(pix_type).reshape(height, width, 4)

    # 2. Выделяем память под итоговый uint8 массив [Height, Width, 3]
    rgb_img = np.zeros((height, width, 3), dtype=pix_type)

    # 3. Достаем каналы из байтов.
    # В упакованном uint32 (BGRA) байты идут строго друг за другом.
    # Сразу переводим в uint8 и масштабируем (<< 8), чтобы получить диапазон 0-65535.

    # Предполагаем стандартный для ToupCam порядок BGRA (байт 0 = B, байт 1 = G, байт 2 = R):
    b = bytes_array[:, :, 0].astype(pix_type)
    g = bytes_array[:, :, 1].astype(pix_type)
    r = bytes_array[:, :, 2].astype(pix_type)

    rgb_img[:, :, 0] = r  # Red
    rgb_img[:, :, 1] = g  # Green
    rgb_img[:, :, 2] = b  # Blue
    return rgb_img


class PatchedCoreViewerLink(QObject):
    """QObject linking events in a napari viewer to events in a CMMCorePlus instance."""

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        core: CMMCorePlus,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._mmc = core
        self.viewer = viewer
        self._mda_handler = _NapariMDAHandler(self._mmc, viewer)
        self._live_timer_id: int | None = None
        self._mda_poll_timer_id: int | None = None

        # Add all core connections to this list.  This makes it easy to disconnect
        # from core when this widget is closed.
        self._connections: list[tuple[PSignalInstance, Callable]] = [
            (self._mmc.events.imageSnapped, self._image_snapped),
            (self._mmc.events.imageSnapped, self._stop_live),
            (self._mmc.events.continuousSequenceAcquisitionStarted, self._start_live),
            (self._mmc.events.sequenceAcquisitionStopped, self._stop_live),
            # (self._mmc.events.exposureChanged, self._restart_live), this connections causes deadlock when exposure changes with live view activated
            (self._mmc.events.configSet, self._restart_live),
            (self._mmc.mda.events.sequenceStarted, self._start_mda_poll),
            (self._mmc.mda.events.sequenceFinished, self._stop_mda_poll),
        ]
        for signal, slot in self._connections:
            signal.connect(slot)

    def cleanup(self, owns: bool = False) -> None:
        """Stop viewer-side activity and, when owned, tear down the core.

        Parameters
        ----------
        owns : bool, default False
            Whether the plugin owns the core. When True, also cancel the
            running MDA (if any) and unload all devices. When False, leave
            the core's MDA and devices alone — the owner is still using it.
        """
        self._stop_live()
        self._stop_mda_poll()
        with contextlib.suppress(Exception):
            self._mmc.stopSequenceAcquisition()
        # MDA cancel: the runner spawns a non-daemon thread that would
        # otherwise block interpreter shutdown and keep exclusive device
        # handles bound. unloadAllDevices: release devices immediately
        # rather than waiting for a non-deterministic GC of the core.
        if owns:
            with contextlib.suppress(Exception):
                self._mmc.mda.cancel()
            with contextlib.suppress(Exception):
                self._mmc.unloadAllDevices()

        for signal, slot in self._connections:
            with contextlib.suppress(Exception):
                signal.disconnect(slot)
        # Clean up temporary files we opened.
        self._mda_handler._cleanup()

    def timerEvent(self, a0: QTimerEvent | None) -> None:
        if a0 is not None and a0.timerId() == self._mda_poll_timer_id:
            self._poll_mda_updates()
        else:
            self._update_viewer()

    def _start_mda_poll(self, *_: object) -> None:
        if self._mda_poll_timer_id is None:
            self._mda_poll_timer_id = self.startTimer(50, Qt.TimerType.PreciseTimer)

    def _stop_mda_poll(self, *_: object) -> None:
        if self._mda_poll_timer_id is not None:
            self.killTimer(self._mda_poll_timer_id)
            self._mda_poll_timer_id = None

    def _poll_mda_updates(self) -> None:
        handler = self._mda_handler
        while handler._viewer_updates:
            handler._update_viewer_dims(handler._viewer_updates.popleft())

    def _image_snapped(self) -> None:
        # Two layers gate the preview update during MDA:
        # - mmc.mda.is_running() is set synchronously by the runner and
        #   covers the start race where _on_mda_started's queued main-thread
        #   handler hasn't yet set _mda_running True.
        # - _mda_handler._mda_running is the latched flag cleared on the
        #   main thread by _on_mda_finished; it covers the symmetric end
        #   race where the runner has already cleared _running and emitted
        #   sequenceFinished but a trailing imageSnapped slot is still
        #   pending.
        if not (self._mmc.mda.is_running() or self._mda_handler._mda_running):
            self._update_viewer(self._mmc.getImage())

    def _start_live(self) -> None:
        interval = 40  # 40 milliseconds = 25 fps
        self._live_timer_id = self.startTimer(interval, Qt.TimerType.PreciseTimer)

    def _stop_live(self) -> None:
        if self._live_timer_id is not None:
            self.killTimer(self._live_timer_id)
            self._live_timer_id = None

    def _restart_live(self, camera: str, exposure: float) -> None:
        if self._live_timer_id:
            self._mmc.stopSequenceAcquisition()
            self._mmc.startContinuousSequenceAcquisition()

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

        if data is not None and len(data.shape) == 2:
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


def apply_mm_live_view_patch(need_to_apply: bool):
    if need_to_apply:
        import napari_micromanager._core_link as core_link
        import napari_micromanager.main_window as mw

        core_link.CoreViewerLink = PatchedCoreViewerLink
        mw.CoreViewerLink = PatchedCoreViewerLink
    else:
        pass
