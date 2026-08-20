import sys

import cv2
import numpy as np
from pymmcore_plus import CMMCorePlus
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class VideoProcessor(QMainWindow):
    def __init__(self, use_mock_camera: bool) -> None:
        super().__init__()

        self.use_mock_camera = use_mock_camera

        self.setWindowTitle("Тест: видео + дифференцирование (Собель)")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Метка для отображения видео
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumSize(320, 240)
        layout.addWidget(self.video_label)

        # Панель управления
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        self.process_btn = QPushButton("Включить обработку (Собель)")
        self.process_btn.setCheckable(True)
        self.process_btn.toggled.connect(self.toggle_processing)
        control_layout.addWidget(self.process_btn)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(100)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        control_layout.addWidget(QLabel("Порог:"))
        control_layout.addWidget(self.threshold_slider)

        self.info_btn = QPushButton("Camera info")
        self.info_btn.clicked.connect(self.show_camera_info)
        control_layout.addWidget(self.info_btn)

        # Таймер для обновления кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Инициализация камеры
        if use_mock_camera:
            self._mmc = CMMCorePlus.instance()
            self._mmc.loadSystemConfiguration()
            self.cam_index = None
        else:
            self.cam_index = self.find_camera()

        self.init_camera(self.cam_index)
        self.init_window()

        self.processing_enabled = False

    def find_camera(self, max_index: int = 10) -> None:
        """Автоматический поиск доступной камеры"""
        print("Поиск доступных камер...")
        available_cameras = []

        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append((i, width, height))
                cap.release()
                print(f"  ✓ Камера {i}: {width}x{height}")
            else:
                print(f"  ✗ Камера {i}: недоступна")

        if not available_cameras:
            print("Ошибка: камеры не найдены!")
            sys.exit(1)

        # Используем первую найденную камеру
        cam_index = available_cameras[0][0]
        print(f"\nCamera with index {cam_index} has been chosen")
        return cam_index

    def init_camera(self, cam_index: int) -> None:
        """Инициализация камеры"""
        if not self.use_mock_camera:
            print(f"Попытка открыть камеру {cam_index} через V4L2...")
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)  # Явно указываем V4L2

            if not self.cap.isOpened():
                print(f"Ошибка: не удалось открыть камеру {cam_index}  через V4L2")
                # Пробуем без указания бэкенда
                self.cap = cv2.VideoCapture(cam_index)
                if not self.cap.isOpened():
                    print("Error: unable to open camera with any backend")
                    sys.exit(1)

            # Попытка установить разрешение
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Проверяем, что камера действительно открыта и читается
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("Предупреждение: камера открыта, но не даёт изображение")
                # Пробуем переинициализировать
                self.cap.release()
                self.cap = cv2.VideoCapture(cam_index)
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    print("Ошибка: камера не даёт изображение")
                    sys.exit(1)

            # Чтение фактического разрешения
            self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Чтение других параметров
            self.cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.cam_format = self.cap.get(cv2.CAP_PROP_FORMAT)

            print(f"Разрешение камеры: {self.cam_width}x{self.cam_height}")
            print(f"FPS: {self.cam_fps}")
            print(f"Индекс камеры: {cam_index}")

            self.timer.start(30)  # мс
        else:
            self._mmc.setExposure(40)
            self._mmc.startContinuousSequenceAcquisition()

            # Чтение фактического разрешения
            self.cam_width = int(self._mmc.getImageWidth())
            self.cam_height = int(self._mmc.getImageHeight())

            # Чтение других параметров
            self.cam_fps = 1000 / self._mmc.getExposure()
            self.cam_format = None

            print(f"Разрешение камеры: {self.cam_width}x{self.cam_height}")
            print(f"FPS: {self.cam_fps}")
            print("Индекс камеры: None")

            self.timer.start(int(self._mmc.getExposure()) or 30)

    def init_window(self) -> None:
        # Устанавливаем размер окна
        window_width = self.cam_width
        window_height = self.cam_height + 100

        screen_geometry = QApplication.primaryScreen().geometry()
        max_width = screen_geometry.width() - 50
        max_height = screen_geometry.height() - 50

        if window_width > max_width or window_height > max_height:
            scale = min(max_width / window_width, max_height / window_height)
            window_width = int(window_width * scale)
            window_height = int(window_height * scale)

        self.resize(window_width, window_height)
        self.video_label.setMinimumSize(self.cam_width, self.cam_height)

    def show_camera_info(self) -> None:
        """Output camera info"""
        if not hasattr(self, "cap") or self.cap is None:
            print("Камера не инициализирована")
            return

        print("\n=== Camera info ===")
        print(f"Индекс: {self.cam_index}")
        print(f"Разрешение: {self.cam_width}x{self.cam_height}")
        print(f"FPS: {self.cam_fps:.2f}")
        print(f"Формат: {self.cam_format}")

        # Дополнительные параметры
        props = {
            "Яркость": cv2.CAP_PROP_BRIGHTNESS,
            "Контраст": cv2.CAP_PROP_CONTRAST,
            "Насыщенность": cv2.CAP_PROP_SATURATION,
            "Оттенок": cv2.CAP_PROP_HUE,
            "Усиление": cv2.CAP_PROP_GAIN,
            "Экспозиция": cv2.CAP_PROP_EXPOSURE,
            "Фокус": cv2.CAP_PROP_FOCUS,
            "Белый баланс": cv2.CAP_PROP_WB_TEMPERATURE,
        }

        for name, prop_id in props.items():
            value = self.cap.get(prop_id)
            if value >= 0:
                print(f"{name}: {value:.2f}")

        print("============================\n")

    def toggle_processing(self, checked: bool) -> None:
        self.processing_enabled = checked
        self.process_btn.setText(
            "Выключить обработку" if checked else "Включить обработку (Собель)"
        )

    def update_frame(self) -> None:
        if self.use_mock_camera:
            frame = self._mmc.getLastImage()
            try:
                bit_depth = int(self._mmc.getImageBitDepth())
            except Exception:
                bit_depth = 16

            shift = max(0, bit_depth - 8)
            frame = (frame >> shift).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            if not hasattr(self, "cap") or self.cap is None:
                return

            ret, frame = self.cap.read()
            if not ret or frame is None:
                return

        if self.processing_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad = np.sqrt(sobelx**2 + sobely**2)
            grad = np.uint8(np.clip(grad, 0, 255))
            threshold = self.threshold_slider.value()
            _, edge = cv2.threshold(grad, threshold, 255, cv2.THRESH_BINARY)
            frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        else:
            pass

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self, "timer"):
            self.timer.stop()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        if self.use_mock_camera:
            self._mmc.stopSequenceAcquisition()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessor(False)
    window.show()
    sys.exit(app.exec())
