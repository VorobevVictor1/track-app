import napari
from src.track.widget import MainWidget

if __name__ == "__main__":
    from mm_camera_patch import apply_mm_live_view_patch

    APPLY_MM_PATCH = False
    # fix grayscale live-view issue
    apply_mm_live_view_patch(APPLY_MM_PATCH)

    # Create a `viewer`
    viewer = napari.Viewer()
    # Instantiate your widget
    my_widg = MainWidget(viewer)

    # Add widget to `viewer`
    viewer.window.add_dock_widget(my_widg)
    napari.run()
