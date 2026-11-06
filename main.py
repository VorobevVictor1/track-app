import napari
from napari.layers import LayerLock
from pymmcore_plus import CMMCorePlus

from pathlib import Path
from tomllib import TOMLDecodeError
from math import pi, sqrt
import numpy as np

from src.track.widget import MainWidget, OBJECTIVE_PX_PER_UM
import src.track.services.app_configuration as app_conf
from src.track.data_models import Objective


def gen_circle_coords(square: float, scale: float) -> np.ndarray:
    """
    Generates the coordinates of the vertices of a square
    circumscribed around a circle of a given area

    Args:
        square (float): Circle area in um^2
        scale (float): scale factor in px/um

    Returns:
        np.ndarray: 2D vertices coordinates
    """
    diam_px = 2 * scale * sqrt(square / pi)
    coords = np.empty((4, 2), dtype=np.float32)
    coords[0] = [1.0, 1.0]  # up_left
    coords[1] = [1.0, 1.0 + diam_px]  # up_right
    coords[2] = [1.0 + diam_px, 1.0 + diam_px]  # down_right
    coords[3] = [1.0 + diam_px, 1.0]  # down_left
    return coords


def points_text_update_cb(event):
    event.source.text.refresh(event.source.features)


if __name__ == "__main__":
    # load config
    try:
        cfg_path = Path("app_config.toml")
        app_conf.config_instance = app_conf.ConfigService.from_file(cfg_path)
    except TOMLDecodeError as e:
        raise TOMLDecodeError(f"Unable to initialize config: {e}")
    print(app_conf.config_instance.settings)

    if app_conf.config_instance.settings.apply_mm_patch:
        from mm_camera_patch import apply_mm_live_view_patch

        # fix grayscale live-view issue
        apply_mm_live_view_patch(app_conf.config_instance.settings.apply_mm_patch)

    # Create a `viewer`
    viewer = napari.Viewer()

    # # for tests
    sh_l = viewer.add_shapes(
        data=gen_circle_coords(962.0, OBJECTIVE_PX_PER_UM[Objective.X100]),
        shape_type="ellipse",
        edge_color="red",
        edge_width=5,
        face_color="#ffffff00",
    )

    text_p = {
        "string": "{index}",
        "size": 30,
        "color": "red",
        "anchor": "upper_right",
        "translation": [10, 40],
        "scaling": True,
    }
    p_l = viewer.add_points(
        size=30, face_color="red", border_color="red", symbol="cross", text=text_p
    )
    p_l.events.features.connect(points_text_update_cb)

    sh_l.locked = LayerLock.DELETION
    p_l.locked = LayerLock.DELETION
    # nmmw = viewer.window.add_plugin_dock_widget('napari-micromanager')
    mmc = CMMCorePlus.instance()

    mmc.loadSystemConfiguration(r"C:\Program Files\Micro-Manager\MMConfig_demo.cfg")
    # mmc.loadSystemConfiguration(r"C:\Program Files\Micro-Manager\touptek.cfg")
    mmc.setAutoShutter(False)

    # Instantiate your widget
    my_widg = MainWidget(viewer, mmc)

    # Add widget to `viewer`
    w = viewer.window.add_dock_widget(
        my_widg, name="COMET-FTA", menu=viewer.window.window_menu
    )
    w.title.close_button.hide()

    napari.run()
