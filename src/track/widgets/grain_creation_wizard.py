import numpy as np
from numpy.typing import NDArray
import cv2
from enum import IntEnum, StrEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWizard,
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
)

from src.track.data_model.grain_cls import Grain, GrainAttributeAssignmentError


class Page(IntEnum):
    INTRO = auto()
    REFL_SURF = auto()
    TRANSM_SURF = auto()
    TRANSM_STACK = auto()
    TRANSM_EDF = auto()
    REFL_EDF = auto()
    SUMMARY = auto()


class FieldName(StrEnum):
    NEED_REFL_EDF = "need_refl_edf"
    NEED_TRANSM_EDF = "need_transm_edf"
    NEED_TRANSM_STACK = "need_transm_stack"
    GRAIN_ID = "grain_id"


class IntroPage(QWizardPage):
    def __init__(self, default_id: int, parent=None):
        super().__init__(parent)
        self.setTitle("Grain creation settings")
        self.setSubTitle("Set grain number and choose images to capture.")

        layout = QVBoxLayout()

        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Grain number:"))
        self.id_edit = QLineEdit(str(default_id))
        id_layout.addWidget(self.id_edit)
        layout.addLayout(id_layout)

        self.chb_refl_edf = QCheckBox("Add reflected EDF image")
        self.chb_transm_stack = QCheckBox("Add z-stack of transmittted images")
        self.chb_transm_edf = QCheckBox("Add transmitted EDF image")

        layout.addWidget(self.chb_refl_edf)
        layout.addWidget(self.chb_transm_stack)
        layout.addWidget(self.chb_transm_edf)

        self.setLayout(layout)

        self.registerField(FieldName.GRAIN_ID, self.id_edit)
        self.registerField(FieldName.NEED_REFL_EDF, self.chb_refl_edf)
        self.registerField(FieldName.NEED_TRANSM_STACK, self.chb_transm_stack)
        self.registerField(FieldName.NEED_TRANSM_EDF, self.chb_transm_edf)


class ImageSnapPage(QWizardPage):
    def __init__(
        self,
        title: str,
        subtitle: str,
        microscope_controller: "CMMCorePlus",
        parent=None,
    ):
        super().__init__(parent)
        self.setTitle(title)
        self.setSubTitle(subtitle)
        self.image_array: NDArray[np.uint8] | NDArray[np.uint16]
        self.microscope = microscope_controller

        layout = QVBoxLayout()

        self.btn_snap = QPushButton("Snap image")
        self.btn_snap.clicked.connect(self.on_snap)
        layout.addWidget(self.btn_snap)

        self.preview_label = QLabel("Preview is empty")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(
            int(self.microscope.getImageWidth() / 8),
            int(self.microscope.getImageHeight() / 8),
        )
        self.preview_label.setStyleSheet(
            "border: 1px solid gray; background-color: #222; color: #aaa;"
        )
        layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

    def on_snap(self):
        self.image_array = self.create_numpy_image()
        new_shape = (
            int(self.image_array.shape[-2] / 8),
            int(self.image_array.shape[-3] / 8),
        )
        preview_img = cv2.resize(self.image_array, new_shape)
        self.update_preview(preview_img)
        self.completeChanged.emit()

    def create_numpy_image(self) -> NDArray[np.uint8] | NDArray[np.uint16]:
        raise NotImplementedError

    def update_preview(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w, ch = arr.shape
        q_img = QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(q_img))

    def isComplete(self) -> bool:
        return hasattr(self, "image_array")


class SurfacePage(ImageSnapPage):
    def __init__(self, type: str, microscope_controller: "CMMCorePlus", parent=None):
        super().__init__(
            f"Snap {type} surface image",
            "Press button to snap image.",
            microscope_controller,
            parent,
        )

    def create_numpy_image(self) -> NDArray[np.uint8] | NDArray[np.uint16]:
        return self.microscope.snap()


class EDFPage(ImageSnapPage):
    def __init__(self, microscope_controller: "CMMCorePlus", parent=None):
        super().__init__(
            "EDF image capture",
            "Press button to create EDF image.",
            microscope_controller,
            parent,
        )

    def create_numpy_image(self) -> np.ndarray:
        return self.microscope.snap()


class SummaryPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Approval page")
        self.setSubTitle("Grain creation process completed.")
        self.setFinalPage(True)

        layout = QVBoxLayout()
        self.info_label = QLabel()
        # self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self.setLayout(layout)

    def initializePage(self):
        wizard: ObjectWizard = self.wizard()

        try:
            wizard.build_object()
        except GrainAttributeAssignmentError as e:
            self.info_label.setText(f"{e}")
        else:
            obj = wizard.created_object

            if obj:
                self.info_label.setText(
                    f"<b>Grain {obj.number} will be created with next properties:</b><br><br>"
                    f"Reflected surface image: <br>"
                )


class ObjectWizard(QWizard):
    def __init__(
        self, default_id: int, microscope_controller: "CMMCorePlus", parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Grain creation wizard")
        self.created_object: Grain | None = None
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.intro_page = IntroPage(default_id)
        self.refl_surf_page = SurfacePage("reflected", microscope_controller)
        self.refl_edf_page = EDFPage(microscope_controller)
        self.transm_surf_page = SurfacePage("transmitted", microscope_controller)
        self.transm_edf_page = EDFPage(microscope_controller)
        self.summary_page = SummaryPage()

        self.setPage(Page.INTRO, self.intro_page)
        self.setPage(Page.REFL_SURF, self.refl_surf_page)
        self.setPage(Page.REFL_EDF, self.refl_edf_page)
        self.setPage(Page.TRANSM_SURF, self.transm_surf_page)
        self.setPage(Page.TRANSM_EDF, self.transm_edf_page)
        self.setPage(Page.SUMMARY, self.summary_page)

    def nextId(self) -> int:
        match self.currentId():
            case Page.INTRO:
                return Page.REFL_SURF
            case Page.REFL_SURF:
                if self.field(FieldName.NEED_REFL_EDF):
                    return Page.REFL_EDF
                else:
                    return Page.TRANSM_SURF
            case Page.REFL_EDF:
                return Page.TRANSM_SURF
            case Page.TRANSM_SURF:
                if self.field(FieldName.NEED_TRANSM_STACK):
                    # return Page.TRANSM_STACK #TODO: implement transm stack capture page
                    return Page.TRANSM_EDF
                if self.field(FieldName.NEED_TRANSM_EDF):
                    return Page.TRANSM_EDF
                else:
                    return Page.SUMMARY
            # case Page.TRANSM_STACK: #TODO: implement transm stack capture page
            #     if self.field(FieldName.NEED_TRANSM_EDF):
            #         return Page.TRANSM_EDF
            #     else:
            #         return Page.SUMMARY
            case Page.TRANSM_EDF:
                return Page.SUMMARY
            case Page.SUMMARY:
                return -1
            case _:  # TODO: implement error page
                return -1

    def build_object(self):
        grain_id = int(self.field(FieldName.GRAIN_ID))
        refl_surf = self.refl_surf_page.image_array
        transm_surf = self.transm_surf_page.image_array

        self.created_object = Grain(
            number=grain_id, refl_surface=refl_surf, transm_surface=transm_surf
        )
