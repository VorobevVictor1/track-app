import cv2
from pymmcore_plus import CMMCorePlus

NUM_IMAGES = 10
mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration()


mmc.startContinuousSequenceAcquisition()
while True:
    if mmc.getRemainingImageCount() > 0:
        image = mmc.getLastImage()
        print(f"Получен кадр размером: {image.shape}")
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("image.png", bgr_image)
        break

print("Остановка захвата...")
mmc.stopSequenceAcquisition()
