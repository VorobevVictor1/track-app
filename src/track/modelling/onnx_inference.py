import cv2
import numpy as np
import onnxruntime as ort

MODEL = r"segm_model\HALtracks_2D.onnx"
SHORT = 2000
CROP = 1800
THRESH = 0.0  # порог по логитам (= 0.5 по вероятности)

_sess_cache = {}


def _session(model=MODEL):
    if model not in _sess_cache:
        _sess_cache[model] = ort.InferenceSession(
            model, providers=["CPUExecutionProvider"]
        )
    return _sess_cache[model]


# ---------- служебные ----------
def _to_gray(a, name):
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name}: ожидается numpy-массив, получен {type(a).__name__}")
    if a.ndim == 3 and a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY)
    elif a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    elif a.ndim != 2:
        raise ValueError(f"{name}: неподдерживаемая форма {a.shape}")
    return a


def _norm(a):
    """uint8 -> /255, uint16 -> /65535 (как np.iinfo(...).max в трейне); float ожидаем в [0,1]."""
    if np.issubdtype(a.dtype, np.integer):
        return (a / np.iinfo(a.dtype).max).astype(np.float32)
    return a.astype(np.float32)


def _resize_short_side(img, size=SHORT):
    h, w = img.shape[:2]
    ow, oh = (size, int(size * h / w)) if w < h else (int(size * w / h), size)
    return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)  # (width, height)!


def _center_crop(img, size=CROP):
    h, w = img.shape[:2]
    t, l = (h - size) // 2, (w - size) // 2
    return img[t : t + size, l : l + size]


def _forward_geometry(h0, w0):
    ow, oh = (SHORT, int(SHORT * h0 / w0)) if w0 < h0 else (int(SHORT * w0 / h0), SHORT)
    t, l = (oh - CROP) // 2, (ow - CROP) // 2
    return (oh, ow), (t, l)


# ---------- основной API ----------
def predict_mask(refl, trans=None, thresh=THRESH, model=MODEL, return_valid=False):
    """
    Инференс одним вызовом.

    Parameters
    ----------
    refl  : np.ndarray (H, W) или (H, W, 3/4), uint8/uint16 — отражённый канал.
    trans : то же, опционально — проходящий канал (как TransFFT в трейне).
            Если None — во все 3 канала сети идёт refl.
    thresh: порог по логитам (0.0 <=> 0.5 по вероятности).
    return_valid: если True, вторым значением возвращает карту (H, W) uint8,
            где 255 — область, которую модель реально видела (окно кропа).

    Returns
    -------
    mask : np.ndarray (H, W) uint8 {0, 255} — того же размера, что вход.
    """
    refl = _to_gray(refl, "refl")
    trans = refl if trans is None else _to_gray(trans, "trans")
    if refl.shape != trans.shape:
        raise ValueError(
            f"размеры refl и trans не совпадают: {refl.shape} vs {trans.shape}"
        )
    h0, w0 = refl.shape

    # препроцессинг ровно как в GrainDataset: resize -> crop -> [0,1] -> CHW
    r = _norm(_center_crop(_resize_short_side(refl)))
    t = _norm(_center_crop(_resize_short_side(trans)))
    x = np.stack([r, t, t], axis=0)[None]  # (1, 3, 1800, 1800)

    # батч: у экспортированной модели он фиксированный (2) — дублируем вход
    sess = _session(model)
    inp = sess.get_inputs()[0]
    batch = inp.shape[0] if isinstance(inp.shape[0], int) else 1
    xb = np.repeat(x, batch, axis=0) if batch > 1 else x

    logits = sess.run(None, {inp.name: xb})[0][0, 0]  # (1800, 1800), берём элемент 0
    m18 = (logits > thresh).astype(np.uint8) * 255

    # обратная геометрия: окно кропа -> канвас (oh, ow) -> исходный (h0, w0)
    (oh, ow), (top, left) = _forward_geometry(h0, w0)
    canvas = np.zeros((oh, ow), np.uint8)
    valid = np.zeros((oh, ow), np.uint8)
    canvas[top : top + CROP, left : left + CROP] = m18
    valid[top : top + CROP, left : left + CROP] = 255
    mask = cv2.resize(canvas, (w0, h0), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(valid, (w0, h0), interpolation=cv2.INTER_NEAREST)

    return (mask, valid) if return_valid else mask / 255  # temp fix for bin mask
