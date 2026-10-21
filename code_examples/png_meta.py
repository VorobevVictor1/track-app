import struct
import zlib


def create_png_text_chunk(keyword: str, text: str) -> bytes:
    """
    Создает байты для стандартного блока tEXt.
    Структура данных tEXt: Ключ (ASCII) + Нулевой байт (\x00) + Текст (ASCII/Latin-1)
    """
    # Преобразуем строки в байты
    keyword_bytes = keyword.encode("latin-1", errors="replace")
    text_bytes = text.encode("latin-1", errors="replace")

    # 1. Тип блока
    chunk_type = b"tEXt"

    # 2. Формируем тело данных (Data)
    chunk_data = keyword_bytes + b"\x00" + text_bytes

    # 3. Считаем длину данных (4 байта, Big-Endian)
    length = struct.pack(">I", len(chunk_data))

    # 4. Считаем CRC-32 (от типа блока и самих данных)
    crc = zlib.crc32(chunk_type + chunk_data)
    crc_bytes = struct.pack(">I", crc)

    # Собираем блок воедино
    return length + chunk_type + chunk_data + crc_bytes


def inject_metadata_into_png(input_path: str, output_path: str, key: str, value: str):
    """
    Читает PNG, находит блок IEND и вставляет перед ним блок tEXt.
    """
    with open(input_path, "rb") as f:
        png_data = f.read()

    # Проверяем сигнатуру PNG (первые 8 байт)
    if png_data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Файл не является валидным PNG")

    # Ищем маркер конца файла IEND. В правильном PNG блок IEND занимает последние 12 байт:
    # 4 байта длины (00 00 00 00) + 4 байта типа (IEND) + 4 байта CRC (AE 42 60 82)
    iend_signature = b"\x00\x00\x00\x00IEND\xaeB`\x82"

    if iend_signature not in png_data:
        raise ValueError("Не найден маркер IEND")

    # Генерируем наш кастомный блок с метаданными
    new_chunk = create_png_text_chunk(key, value)

    # Разрезаем файл прямо перед IEND и вставляем наш блок
    parts = png_data.split(iend_signature)
    # Если внутри были другие упоминания байт (маловероятно, но для надежности берем последний элемент)
    main_body = iend_signature.join(parts[:-1])

    final_png = main_body + new_chunk + iend_signature

    with open(output_path, "wb") as f:
        f.write(final_png)
    print(f"Метаданные успешно внедрены в {output_path}")


def read_metadata_from_png(file_path: str, target_key: str) -> str:
    with open(file_path, "rb") as f:
        # Пропускаем сигнатуру
        f.seek(8)

        while True:
            # Читаем длину блока (4 байта)
            length_bytes = f.read(4)
            if not length_bytes or len(length_bytes) < 4:
                break

            length = struct.unpack(">I", length_bytes)[0]
            # Читаем тип блока (4 байта)
            chunk_type = f.read(4)

            # Если это блок tEXt, проверяем ключ
            if chunk_type == b"tEXt":
                data = f.read(length)
                # Данные разделены нулевым байтом: Ключ\x00Значение
                parts = data.split(b"\x00", 1)
                if len(parts) == 2:
                    key = parts[0].decode("latin-1")
                    if key == target_key:
                        f.seek(4, 1)  # Пропускаем CRC этого блока
                        return parts[1].decode("latin-1")
                # Читаем CRC (4 байта) и переходим дальше, если ключ не тот
                f.seek(4, 1)
            elif chunk_type == b"IEND":
                break
            else:
                # Пропускаем данные блока + 4 байта CRC
                f.seek(length + 4, 1)

    return ""


# --- Пример использования ---
# Допустим, у вас есть файл 'app_screen.png'
# Мы запишем в него JSON-строку с конфигурацией слоя измерений
import json

measurement_layer = {
    "layer_id": "measurements_v1",
    "scale": "1.5",
    "elements": [
        {"type": "line", "start": [10, 20], "end": [100, 200], "label": "4.5m"},
        {"type": "point", "coords": [50, 50], "label": "Anchor A"},
    ],
}

# Превращаем конфиг в строку
metadata_json = json.dumps(measurement_layer)

# Внедряем данные
inject_metadata_into_png(
    input_path="app_screen.png",
    output_path="app_screen_with_meta.png",
    key="MeasurementLayer",
    value=metadata_json,
)

saved_meta = read_metadata_from_png("app_screen_with_meta.png", "MeasurementLayer")
if saved_meta:
    print("Найдены метаданные:", json.loads(saved_meta))
    print(saved_meta == metadata_json)
