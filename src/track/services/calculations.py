from ..data_models import CalculationRequest, CalculationResult
import numpy as np
from skimage.measure import label


class CalculationService:
    def process_calc_request(self, request: CalculationRequest) -> CalculationResult:
        num_of_tracks = self.tracks_calc(request.tracks_map, request.roi_mask)

        density = self.density_calc(num_of_tracks, request.area_px, request.px_per_um)

        return CalculationResult(num_of_tracks=num_of_tracks, tracks_density=density)

    def density_calc(self, num_of_tracks: int, area_px: float, scale: float) -> float:
        """Calculates tracks density"""
        if area_px != 0:
            tracks_density_by_pixel = num_of_tracks / area_px
            tracks_density = tracks_density_by_pixel * scale**2
        else:
            tracks_density = 0
        return round(tracks_density, 5)

    def tracks_calc(
        self, tracks_labels: np.ndarray, roi_mask: np.ndarray | None
    ) -> int:
        """
        Counts number of tracks in ROI

        Args:
            tracks_labels (np.ndarray): tracks labeled array. background pixels = 0,
            tracks pixels are non zero positive values
            roi_mask (np.ndarray): binary roi mask pixels in roi = 1

        Returns:
            int: num of tracks (not tracks pixels) in ROI
        """
        if roi_mask is not None:
            tracks_in_roi = tracks_labels * roi_mask
        else:
            tracks_in_roi = tracks_labels
        _, num_of_tracks = label(
            label_image=tracks_in_roi, background=0, return_num=True, connectivity=2
        )
        return num_of_tracks
