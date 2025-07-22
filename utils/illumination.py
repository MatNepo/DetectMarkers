import cv2
import numpy as np
from detectors.aruco.params import custom_aruco_params
from detectors.aruco.generate import load_custom_aruco_dict, get_streams
from utils.distance_counter import get_stream_id
import pandas as pd
import os

file_path = "data/custom_dicts/DICT_5X5_10000.yml"
aruco_dict = load_custom_aruco_dict(filename=file_path)
parameters = custom_aruco_params(print_default_values=False, print_custom_values=False)


def update_brightness_table(marker_id, brightness, filename='data.csv'):
    # Проверяем, существует ли файл таблицы
    if os.path.exists(filename):
        # Загружаем существующую таблицу
        df = pd.read_excel(filename, engine='openpyxl')
    else:
        # Создаем новую таблицу, если файла нет
        df = pd.DataFrame(columns=['ID', 'Освещённость'])

    # Проверяем, существует ли уже строка с данным ID
    if marker_id in df['ID'].values:
        # Обновляем освещённость для существующего ID
        df.loc[df['ID'] == marker_id, 'Освещённость'] = brightness
    else:
        # Добавляем новую строку с ID и освещённостью
        new_row = pd.DataFrame({'ID': [marker_id], 'Освещённость': [brightness]})
        df = pd.concat([df, new_row], ignore_index=True, engine='openpyxl')

    # Сохраняем таблицу
    df.to_csv(filename, index=False)


def calculate_brightness(frame, corners):
    padding = 2
    x1 = int(min(corner[0] for corner in corners[0])) - padding
    y1 = int(min(corner[1] for corner in corners[0])) - padding
    x2 = int(max(corner[0] for corner in corners[0])) + padding
    y2 = int(max(corner[1] for corner in corners[0])) + padding
    roi = frame[y1:y2, x1:x2]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_roi)
    return brightness


# Пример: используйте camera_X для имён камер
# Замените camera_X на свои реальные имена камер/локаций
camera_name = input('Camera name: ')
stream = get_stream_id(camera_name=camera_name)
print(stream)
cap = cv2.VideoCapture(stream)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_candidates = detector.detectMarkers(frame)

    if ids is not None:
        for i, corner in enumerate(corners):
            # Расчет яркости области вокруг маркера
            brightness = calculate_brightness(frame, corner)
            marker_id = ids[i][0]
            update_brightness_table(marker_id=marker_id, brightness=brightness)
            # print(f'Освещённость (средняя яркость) для маркера {ids[i][0]}: {brightness}')
            text_position = (int(corner[0][0][0]), int(corner[0][0][1] - 10))
            cv2.putText(frame, f"ID {marker_id}: {brightness:.2f}", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == (ord('q') or ord('й')):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
