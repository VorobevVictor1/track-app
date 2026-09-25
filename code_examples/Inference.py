import cv2
import numpy as np
import onnxruntime as ort

MODEL = "HALtracks_2D.onnx"
SHORT = 2000  # как в T.Resize(2000)
CROP = 1800  # как в T.CenterCrop(1800)
BATCH = 2  # фиксированный батч экспортированной модели
THRESH = 0.0  # порог по логитам (= 0.5 по вероятности)


def forward_geometry(h0, w0, short=SHORT, crop=CROP):
    """Что именно сделали resize+center_crop: размер после resize и окно кропа."""
    if w0 < h0:
        ow, oh = short, int(short * h0 / w0)
    else:
        ow, oh = int(short * w0 / h0), short
    top, left = (oh - crop) // 2, (ow - crop) // 2
    return (oh, ow), (top, left)


def mask_to_orig(mask, orig_hw):
    """Маска 1800x1800 -> (маска в исходных координатах, карта 'модель сюда смотрела')."""
    h0, w0 = orig_hw
    (oh, ow), (top, left) = forward_geometry(h0, w0)

    canvas = np.zeros((oh, ow), np.uint8)
    valid = np.zeros((oh, ow), np.uint8)
    canvas[top : top + CROP, left : left + CROP] = mask
    valid[top : top + CROP, left : left + CROP] = 255

    m = cv2.resize(canvas, (w0, h0), interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(valid, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return m, v


def overlay(orig_gray, mask_orig, valid_orig, alpha=0.45):
    bgr = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(bgr)
    tint[:] = (0, 0, 255)  # красный, BGR
    m = mask_orig > 0
    bgr[m] = cv2.addWeighted(bgr, 1 - alpha, tint, alpha, 0)[m]

    nv = valid_orig == 0  # модель там не видела
    bgr[nv] = (bgr[nv] * 0.35).astype(np.uint8)  # затемнить
    cnts, _ = cv2.findContours(valid_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, cnts, -1, (0, 255, 255), 3)  # жёлтая рамка окна
    return bgr


# ---------- чтение ----------
def imread_gray(path):
    """IMREAD_UNCHANGED сохраняет uint16; дефолтный imread съедал бы его до uint8."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:  # цветное -> grayscale, как convert("L")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ---------- T.Resize(2000): меньшая сторона -> 2000 ----------
def resize_short_side(img, size=SHORT):
    h, w = img.shape[:2]
    if w < h:  # формула torchvision, int() = truncation
        ow, oh = size, int(size * h / w)
    else:
        ow, oh = int(size * w / h), size
    return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)  # (width, height)!


# ---------- T.CenterCrop(1800) ----------
def center_crop(img, size=CROP):
    h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2  # ровно как torchvision
    return img[top : top + size, left : left + size]


# ---------- нормализация как в open_as_array ----------
def to_float(img):
    return (img / np.iinfo(img.dtype).max).astype(np.float32)


# ---------- полный препроцессинг одного снимка -> (3, 1800, 1800) ----------
def preprocess(refl_path, trans_path=None):
    refl = to_float(center_crop(resize_short_side(imread_gray(refl_path))))
    trans = (
        to_float(center_crop(resize_short_side(imread_gray(trans_path))))
        if trans_path
        else refl
    )
    return np.stack([refl, trans, trans], axis=0)  # CHW: refl, trans, trans


# ---------- инференс ----------
def main(refl_path, trans_path=None, out_prefix="pred"):
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("input:", inp.name, inp.shape, inp.type)
    assert tuple(inp.shape) == (
        BATCH,
        3,
        CROP,
        CROP,
    ), f"ожидается фикс. вход {(BATCH, 3, CROP, CROP)}, получен {inp.shape}"

    x = preprocess(refl_path, trans_path)[None]  # (1, 3, 1800, 1800)
    xb = np.repeat(x, BATCH, axis=0)  # (2, 3, 1800, 1800) — дубль

    out = sess.run(None, {inp.name: xb})[0]  # (2, 1, 1800, 1800)
    logits = out[0, 0]  # первый элемент батча

    mask = (logits > THRESH).astype(np.uint8) * 255
    cv2.imwrite(f"{out_prefix}_mask.png", mask)

    # --- маска и оверлей в исходном размере фото ---
    orig = imread_gray(refl_path)
    mask_orig = cv2.resize(
        mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST
    )  # только NEAREST!
    cv2.imwrite(f"{out_prefix}_mask_orig.png", mask_orig)

    bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    tint = bgr.copy()
    tint[:] = (0, 0, 255)  # красный в BGR
    m = mask_orig > 0
    bgr[m] = cv2.addWeighted(bgr, 0.55, tint, 0.45, 0)[m]
    cv2.imwrite(f"{out_prefix}_overlay.png", bgr)

    print(
        f"покрытие: {m.mean():.2%}, logits min/max: {logits.min():.2f}/{logits.max():.2f}"
    )

    orig = imread_gray(refl_path)
    mask_orig, valid_orig = mask_to_orig(mask, orig.shape[:2])
    cv2.imwrite(f"{out_prefix}_mask_orig.png", mask_orig)
    cv2.imwrite(f"{out_prefix}_overlay.png", overlay(orig, mask_orig, valid_orig))

    # КОНТРОЛЬНЫЙ кадр: оверлей на том, что реально видела модель (всегда должен сходиться)
    view = center_crop(resize_short_side(orig))
    vw = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    vw[mask > 0] = cv2.addWeighted(vw, 0.55, np.full_like(vw, (0, 0, 255)), 0.45, 0)[
        mask > 0
    ]
    cv2.imwrite(f"{out_prefix}_overlay_view1800.png", vw)


if __name__ == "__main__":
    main(
        r"C:\Users\vivor\Desktop\Test-for-VV\1b.jpg",
        r"C:\Users\vivor\Desktop\Test-for-VV\1a.jpg",
    )
    # с парой каналов, как в трейне:
    # main("Durango_1_Grain01_ReflStackFlat.tif", "Durango_1_Grain01_TransFFT.tif")
