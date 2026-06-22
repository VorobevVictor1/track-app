import cv2
import sys

def check_camera(camera_index=0):
    """Проверка подключения USB камеры"""
    print(f"Попытка открытия камеры с индексом {camera_index}...")
    
    # Открываем камеру
    cap = cv2.VideoCapture(camera_index)
    
    # Проверяем, открылась ли камера
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть камеру {camera_index}")
        return False
    
    # Получаем информацию о камере
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Камера успешно открыта!")
    print(f"  Разрешение: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"\nНажмите 'q' для выхода")
    
    # Цикл захвата и отображения видео
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break
        
        # Отображаем кадр
        cv2.imshow(f'Camera {camera_index} - Press Q to quit', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    print("Камера закрыта")
    return True

if __name__ == "__main__":
    # Индекс камеры можно передать как аргумент (по умолчанию 0)
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    try:
        check_camera(camera_idx)
    except Exception as e:
        print(f"Произошла ошибка: {e}")