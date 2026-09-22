import cv2
import numpy as np
from napari_builtins._measure_shapes import ellipse_area, polygon_area, rectangle_area
from segm_model.onnx_inference import predict_mask
from skimage.filters import threshold_isodata
from skimage.measure import label
from skimage.morphology import closing, opening


def measure_area(shape_data_list, shape_type_list):
    vertices = shape_data_list[0]
    shape_type = shape_type_list[0]
    area = None
    if shape_type == "polygon":
        area = polygon_area(vertices)
    elif shape_type == "rectangle":
        area = rectangle_area(vertices)
    elif shape_type == "ellipse":
        area = ellipse_area(vertices)
    elif shape_type == "path" or shape_type == "line":
        area = 0
    return area


def background_correction(img, sigma, gray_level):
    if len(img.shape) > 2:
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = img.astype(np.uint8)

    # Create a low-pass filtered version
    low_pass = cv2.GaussianBlur(grayscale, (0, 0), sigma)

    # Subtract low frequencies from the original image
    # We add 128 to shift the neutral gray level to the center
    high_pass = cv2.addWeighted(grayscale, 1.0, low_pass, -1.0, gray_level)

    return high_pass


def features_exctraction(img):
    # calculates negative feature map: features pixels = 1 and background = 0
    # _, feature_map = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)
    iso_thresh = threshold_isodata(img)
    _, feature_map = cv2.threshold(img, iso_thresh, 1, cv2.THRESH_BINARY_INV)
    return feature_map


def morph_filtering(img, kernel_size=5, kernel_shape="square"):
    "Filter thresholding noise"
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = opening(img, kernel)
    closed = closing(opened, kernel)
    return closed


def segm_postprocessing(labels: np.ndarray, area: float):
    "Counting etc"
    if area is not None:
        num_of_tracks = np.count_nonzero(np.unique(labels))
        tracks_density_by_pixel = num_of_tracks / area
    else:
        tracks_density_by_pixel = 0
    return tracks_density_by_pixel


def instance_segment(img_transmitted, img_reflected, mask, sigma=25, gray_level=128):
    if img_transmitted.shape[:2] != img_reflected.shape[:2]:
        return None

    # TODO auto selection of gray level
    # returns high pass img with flat background level
    img_transmitted_hp = background_correction(img_transmitted, sigma, gray_level)
    img_reflected_hp = background_correction(img_reflected, sigma, gray_level)

    transmitted_features = features_exctraction(img_transmitted_hp)
    reflected_features = features_exctraction(img_reflected_hp)

    # coincedence mapping
    res_binary = transmitted_features * reflected_features
    # ROI mask application
    if mask is not None:
        res_binary = res_binary * mask

    filtered_binary = morph_filtering(res_binary)

    labeled = label(filtered_binary).astype(np.uint16)

    return labeled


def segm_with_nn(img_transmitted, img_reflected, mask, threshold):
    res_binary = predict_mask(img_reflected, img_transmitted, threshold)

    # ROI mask application
    if mask is not None:
        res_binary = res_binary * mask

    labeled = label(res_binary).astype(np.uint16)

    return labeled


# def flat_background(img: np.ndarray, cropping_window_size: int) -> "naptypes.ImageData":
#     if len(img.shape) > 2:
#         grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     else:
#         grayscale = img.astype(np.uint8)
#     fourier_tr = np.fft.fft2(grayscale)
#     shifted_ft = np.fft.fftshift(fourier_tr)
#     rows, cols = shifted_ft.shape
#     crow, ccol = rows//2, cols//2
#     # 3. Create a Bandpass Mask
#     mask = np.zeros((rows, cols), np.uint8)

#     # Generate a grid of distances from the center
#     y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
#     distance = np.sqrt(x*x + y*y)

#     # Keep only frequencies between low_cutoff and high_cutoff
#     bandpass_area = (distance >= cropping_window_size)
#     mask[bandpass_area] = 1

#     # 4. Apply mask and inverse DFT
#     fshift = shifted_ft * mask
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = np.real(img_back)
#     return img_back

# def sobel_filter(img: np.ndarray, threshold: float) -> tuple["naptypes.LabelsData", "np.float64"]:
#     sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#     grad = np.sqrt(sobelx**2 + sobely**2)
#     grad = np.uint8(np.clip(grad, 0, 255))
#     _, edge = cv2.threshold(grad, threshold, 255, cv2.THRESH_BINARY)
#     props = regionprops(edge)
#     area = np.float64(sum(p.area for p in props))
#     return edge, area

# if __name__=='__main__':
#     img1 = imread(r"C:\Users\vivor\Downloads\Test-for-VV\1a.jpg")
#     img2 = imread(r"C:\Users\vivor\Downloads\Test-for-VV\1b.jpg")
#     labels = instance_segment(img1, img2, None)
#     print(segm_postprocessing(labels))
