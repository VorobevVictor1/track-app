from cv2 import COLOR_RGB2BGR, cvtColor, imencode
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from pathlib import Path

def save_img(data: 'np.ndarray', path: 'Path', ext: str):
    data_bgr = cvtColor(data, COLOR_RGB2BGR)
    res, buffer = imencode(ext, data_bgr)
    if res:
        with open(path, 'wb') as f:
            f.write(buffer)
    else:
        raise Exception('Unable to save image')
