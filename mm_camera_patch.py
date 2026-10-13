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


def apply_mm_live_view_patch(need_to_apply: bool):
    if need_to_apply:
        import napari_micromanager._core_link as core_link

        core_link.CoreViewerLink._update_viewer = _update_viewer
    else:
        pass
