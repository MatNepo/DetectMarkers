import math
import csv


# Исходные данные 12.5 cm
camera_height = 3  # высота камеры, например, 3 метра
# Пример структуры: замените camera_X_Y на свои реальные имена камер/локаций и маркеров
markers_dict = {
    'camera_1_0': 3,
    'camera_1_1': 7,
    'camera_1_2': 6,
    'camera_1_10': 1,
    'camera_1_20': 15,
    'camera_2_11': 2,
    'camera_2_9249': 2,
    'camera_3_14': 12,
    'camera_3_15': 4.5,
    'camera_4_12': 10,
    'camera_4_20': 30,
    'camera_5_16': 2,
    # Добавьте свои камеры/маркеры по аналогии
}  # marker_id : steps


def calculate_marker_data(markers=None, camera_height=3, save_to_table=False, print_to_terminal=False):
    if markers is None:
        markers = markers_dict
    results = {}

    coef = 0.795

    # Открываем файл для записи, если включён флаг save_to_table
    if save_to_table:
        with open("data/marker_data.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Marker ID", "Distance (m)", "Angle (degrees)"])

            # Заполняем данные для каждой метки
            for marker_id, steps in markers.items():
                steps = steps * coef
                distance = math.sqrt(camera_height ** 2 + steps ** 2)  # расстояние по прямой
                angle = math.degrees(math.atan(camera_height / steps))  # угол наклона камеры

                results[marker_id] = {"distance": distance, "angle": angle}
                writer.writerow([marker_id, distance, angle])

                # Выводим данные в терминал, если включён флаг
                if print_to_terminal:
                    print(f"Marker {marker_id}: Distance = {distance:.2f} m, Angle = {angle:.2f}°")

    else:
        for marker_id, steps in markers.items():
            steps = steps * coef
            distance = math.sqrt(camera_height ** 2 + steps ** 2)
            angle = math.degrees(math.atan(camera_height / steps))
            results[marker_id] = {"distance": distance, "angle": angle}

            # Выводим данные в терминал, если включён флаг
            if print_to_terminal:
                print(f"Marker {marker_id}: Distance = {distance:.2f} m, Angle = {angle:.2f}°")

    # Если вывод в терминал выключен, возвращаем словарь с результатами
    if not print_to_terminal:
        return results


if __name__ == "__main__":
    calculate_marker_data(markers=markers_dict,
                          camera_height=3,
                          save_to_table=False,
                          print_to_terminal=True
                          )
