import numpy as np
import numpy.typing as npt
from .data_models import ROI


class GrainAttributeAssignmentError(ValueError, TypeError):
    pass


class Grain:
    def __init__(
        self,
        number: int,
        refl_surface: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
        transm_surface: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
        track_pits_mask: npt.NDArray[np.uint8] | None = None,
        roi_list: list[ROI] | None = None,
        track_points_array: npt.NDArray[np.float64] | None = None,
        transm_stack: npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | None = None,
        refl_edf: npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | None = None,
        transm_edf: npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | None = None,
    ):
        self.number = number

        self._check_imgs_size_equality(refl_surface, transm_surface)

        # grain must has this properties
        self.refl_surface_image = refl_surface
        self.transm_surface_image = transm_surface

        # this properties propably will be setted some time letter
        self.track_pits_mask = (
            track_pits_mask
            if track_pits_mask is not None
            else np.zeros(transm_surface.shape[0:2], dtype=np.uint8)
        )
        self.roi_list = roi_list if roi_list is not None else []
        self.track_points_array = (
            track_points_array
            if track_points_array is not None
            else np.zeros((0, 2), dtype=np.float64)
        )

        # this values may be newer setted
        if transm_stack is not None:
            self.transm_stack = transm_stack
        if refl_edf is not None:
            self.refl_edf = refl_edf
        if transm_edf is not None:
            self.transm_edf = transm_edf

    @staticmethod
    def _check_imgs_size_equality(
        refl_surface: npt.NDArray[...],
        transm_surface: npt.NDArray[...],
    ):
        if refl_surface.shape != transm_surface.shape:
            err_str = (
                f"Grain creation failed. Images expected to have the same shape. "
                f"Instead given reflected: {refl_surface.shape}, transmitted: {transm_surface.shape}"
            )
            raise ValueError(err_str)

    @staticmethod
    def _check_source_image_input(img: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]):
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"Source image must be 2D RGB. Got instead shape: {img.shape}"
            )
        if img.dtype != np.uint8 and img.dtype != np.uint16:
            raise TypeError(
                f"Source image has incorrect dtype: {img.dtype}. One of {np.uint8 | np.uint16} expected"
            )
        return img

    @property
    def refl_surface_image(self):
        return self._refl_surface_image

    @refl_surface_image.setter
    def refl_surface_image(self, img: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]):
        if not hasattr(self, "_refl_surface_image"):
            try:
                self._refl_surface_image = self._check_source_image_input(img)
            except (ValueError, TypeError) as exc:
                err = GrainAttributeAssignmentError(
                    "Reflected light surface image assignmnent failed."
                )
                err.add_note(str(exc))
                raise err from None
        else:
            err = (
                "Surface image can be assigned only once. "
                "Create new grain object to assign new surface image."
            )
            raise AttributeError(err)

    @property
    def transm_surface_image(self):
        return self._transm_surface_image

    @transm_surface_image.setter
    def transm_surface_image(self, img: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]):
        if not hasattr(self, "_transm_surface_image"):
            try:
                self._transm_surface_image = self._check_source_image_input(img)
            except (ValueError, TypeError) as exc:
                err = GrainAttributeAssignmentError(
                    "Transmitted light surface image assignmnent failed."
                )
                err.add_note(str(exc))
                raise err from None
        else:
            err = (
                "Surface image can be assigned only once. "
                "Create new grain object to assign new surface image."
            )
            raise AttributeError(err)

    @property
    def track_pits_mask(self):
        return self._track_pits_mask

    def _check_mask_input(self, mask):
        if mask.ndim != 2:
            raise ValueError(
                f"Track pits mask must be 2D array. Got instead ndim: {mask.ndim}"
            )
        if mask.shape != self.refl_surface_image.shape[0:2]:
            err_str = (
                f"Track pits mask must have the same shape as the surface images. "
                f"Instead given track pits mask: {mask.shape}, images: {self.refl_surface_image.shape}"
            )
            raise ValueError(err_str)
        if mask.dtype != np.uint8:
            raise TypeError(
                f"Mask has incorrect dtype: {mask.dtype}. Expected {np.uint8}."
            )
        return mask

    @track_pits_mask.setter
    def track_pits_mask(self, mask: npt.NDArray[np.uint8]):
        try:
            self._track_pits_mask = self._check_mask_input(mask)
        except (ValueError, TypeError) as exc:
            err = GrainAttributeAssignmentError("Track pits mask assignmnent failed.")
            err.add_note(str(exc))
            raise err from None

    @staticmethod
    def _check_roi_input(roi: ROI):
        if not isinstance(roi, ROI):
            raise TypeError(f"Expected ROI object. Got instead {type(roi).__name__}")

    @property
    def roi_list(self):
        return tuple(self._roi_list)

    @roi_list.setter
    def roi_list(self, rois: list[ROI] | tuple[ROI, ...]):
        if not isinstance(rois, (tuple, list)):
            raise GrainAttributeAssignmentError(
                f"Roi list assignmnent failed: must be a list or tuple. Got instead {type(rois).__name__}"
            )

        for i, item in enumerate(rois):
            try:
                self._check_roi_input(item)
            except TypeError as exc:
                err = GrainAttributeAssignmentError(
                    f"ROI list assignmnent failed. Element with index {i} has wrong type."
                )
                err.add_note(str(exc))
                raise err from None

        # save shallow copy of input.
        # NOTE: ROI is a named tuple. It has a nd.array with vertices inside
        self._roi_list = list(rois)

    def add_roi(self, roi: ROI):
        self._check_roi_input(roi)
        self._roi_list.append(roi)

    def remove_roi(self, index: int = -1) -> ROI:
        return self._roi_list.pop(index)

    @property
    def track_points_array(self):
        return self._track_points_array

    @staticmethod
    def _check_track_points_input(points_array: npt.NDArray[np.float64]):
        if points_array.ndim != 2:
            raise ValueError(
                f"Track points must be a 2D array. Got instead {points_array.ndim}"
            )
        if points_array.shape[-1] != 2:
            raise ValueError(
                f"Points array must have size (N, 2). Got instead {points_array.shape}"
            )
        if points_array.dtype != np.float64:
            raise TypeError(
                f"Points array must has dtype: {np.float64}. Got instead: {points_array.dtype}"
            )
        return points_array

    @track_points_array.setter
    def track_points_array(self, points_array: npt.NDArray[np.float64]):
        try:
            self._track_points_array = self._check_track_points_input(points_array)
        except (ValueError, TypeError) as exc:
            err = GrainAttributeAssignmentError("Track points array assignment failed.")
            err.add_note(str(exc))
            raise err from None

    def _check_stack_imgs_input(
        self, img_stack: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
    ):
        if img_stack.ndim != 4:
            raise ValueError(
                f"Image stack stack must be a 4D array. Got instead: {img_stack.ndim}"
            )
        if img_stack.shape[-1] != 3:
            raise ValueError(
                f"Images in stack must be RGB. Last dimension got instead: {img_stack.shape[-1]}"
            )
        if img_stack.shape[1:] != self.refl_surface_image.shape:
            err_str = (
                f"Images in stack must have the same shape as the surface images. "
                f"Instead given shape in stack: {img_stack.shape[1:]}, surface: {self.refl_surface_image.shape}"
            )
            raise ValueError(err_str)
        if img_stack.dtype not in (np.uint8, np.uint16):
            raise TypeError(
                f"Stack images has incorrect dtype: {img_stack.dtype}. One of {np.uint8 | np.uint16} expected"
            )
        return img_stack

    @property
    def transm_stack(self):
        return self._transm_stack

    @transm_stack.setter
    def transm_stack(self, img_stack: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]):
        try:
            self._transm_stack = self._check_stack_imgs_input(img_stack)
        except (ValueError, TypeError) as exc:
            err = GrainAttributeAssignmentError(
                "Transmitted images stack assignment failed."
            )
            err.add_note(str(exc))
            raise err from None

    @property
    def refl_edf(self):
        return self._refl_edf

    @refl_edf.setter
    def refl_edf(self, img: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]):
        try:
            self._refl_edf = self._check_source_image_input(img)
        except (ValueError, TypeError) as exc:
            err = GrainAttributeAssignmentError(
                "Reflected light EDF image assignmnent failed."
            )
            err.add_note(str(exc))
            raise err from None

    @property
    def transm_edf(self):
        return self._transm_edf

    @transm_edf.setter
    def transm_edf(self, img: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]):
        try:
            self._transm_edf = self._check_source_image_input(img)
        except (ValueError, TypeError) as exc:
            err = GrainAttributeAssignmentError(
                "Transmitted light EDF image assignmnent failed."
            )
            err.add_note(str(exc))
            raise err from None
