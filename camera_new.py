import sys

import toupcam

# Поиск камер
cameras = toupcam.Toupcam.EnumV2()
if not cameras:
    print("Камеры не найдены!")
    sys.exit()

print(f"Найдена камера: {cameras[0].displayname}")
print(f"Модель: {cameras[0].model.name}")

# Открытие камеры
cam = toupcam.Toupcam.Open(cameras[0].id)
if cam is None:
    print("Unable to open camera")
    sys.exit()

# Получение размера
width, height = cam.get_Size()
print(f"Resolution: {width}x{height}")

# Закрытие
cam.Close()
print("Test completed!")
