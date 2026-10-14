import napari
from src.track.widget import MainWidget

if __name__ == "__main__":

    APPLY_MM_PATCH = False
    if APPLY_MM_PATCH:
        from mm_camera_patch import apply_mm_live_view_patch
        # fix grayscale live-view issue
        apply_mm_live_view_patch(APPLY_MM_PATCH)

    # Create a `viewer`
    viewer = napari.Viewer()
    # Instantiate your widget
    my_widg = MainWidget(viewer)
    
    #for tests
    # viewer.add_shapes(face_color='#ffffff00')
    # viewer.add_points(size=30)

    # Add widget to `viewer`
    viewer.window.add_dock_widget(my_widg)
    napari.run()
