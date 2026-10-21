import napari
import numpy as np
from magicgui import magicgui

# 1. Инициализируем Viewer
viewer = napari.Viewer()

# 2. Создаем пустой слой для кастомных линий
# Задаем базовые свойства по умолчанию, чтобы napari знал их структуру
initial_properties = {
    "line_type": ["Border"],  # Доступные типы
    "object_id": ["0"],
}

# Маппинг цветов для типов линий
COLOR_CYCLE = {"Border": "crimson", "Path": "cyan", "Danger Zone": "yellow"}

shapes_layer = viewer.add_shapes(
    name="Custom Lines",
    shape_type="line",
    properties=initial_properties,
    edge_color="line_type",
    edge_width=4,
    text={
        "string": "{line_type} [ID: {object_id}]",
        "size": 10,
        "color": "white",
        "translation": [0, -15],  # Смещаем текст чуть выше линии
    },
)

# Очищаем начальную фейковую геометрию, оставляя структуру свойств
shapes_layer.data = []


# 3. Создаем GUI-виджет для управления свойствами "на лету"
@magicgui(
    auto_call=True,  # Виджет мгновенно реагирует на изменения в интерфейсе
    line_type={"choices": ["Border", "Path", "Danger Zone"], "label": "Тип линии:"},
    object_id={"label": "ID Объекта:", "value": "1"},
)
def properties_widget(line_type: str, object_id: str):
    """
    Этот код выполняется каждый раз, когда пользователь
    меняет значения в выпадающем списке или текстовом поле.
    """
    # Обновляем свойства для СЛЕДУЮЩЕЙ нарисованной фигуры
    shapes_layer.current_properties = {
        "line_type": np.array([line_type], dtype=object),
        "object_id": np.array([object_id], dtype=object),
    }


# 4. Добавляем созданный виджет в правый сайдбар napari
viewer.window.add_dock_widget(properties_widget, area="right")

# Включаем режим рисования линий сразу при старте для удобства
shapes_layer.mode = "add_line"

# Запуск приложения
napari.run()
