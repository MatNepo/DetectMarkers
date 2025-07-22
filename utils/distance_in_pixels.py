import cv2
import numpy as np

from detectors.aruco.params import custom_aruco_params
from detectors.aruco.generate import load_custom_aruco_dict, get_streams

custom_dict_path = "data/custom_dicts/DICT_5X5_10000.yml"
aruco_dict = load_custom_aruco_dict(custom_dict_path)
aruco_params = custom_aruco_params(print_default_values=False, print_custom_values=False)

internal_streams, external_streams = get_streams()
streams = internal_streams
print("Доступные камеры (пример):")
for camera in streams:
    print(f"- {camera}")
# Замените camera_X на свои реальные имена камер/локаций
camera_name = input("Введите название камеры для записи: ").lower()
stream = streams[camera_name]

# Открытие камеры
cap = cv2.VideoCapture(stream)

ret, frame = cap.read()
print(frame.shape[1], frame.shape[0])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детектирование маркеров
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected_candidates = detector.detectMarkers(frame)

    if len(corners) > 0:
        # Рисуем маркеры
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Размеры маркеров
        for i, corner in enumerate(corners):
            # Получаем координаты углов маркера
            top_left = corner[0][0]  # Топ-левый угол
            top_right = corner[0][1]  # Топ-правый угол
            bottom_right = corner[0][2]  # Боттом-правый угол
            bottom_left = corner[0][3]  # Боттом-левый угол

            # Вычисляем размеры маркера
            width = np.linalg.norm(top_left - top_right)  # Расстояние между левым и правым углом
            height = np.linalg.norm(top_left - bottom_left)  # Расстояние между верхним и нижним углом
            # print(f"Размер маркера {ids[i][0]}: ширина={width:.2f} пикселей, высота={height:.2f} пикселей")

            # Центр маркера (среднее положение всех углов)
            center = np.mean(corner[0], axis=0)

            # Центр кадра
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

            # Расстояние от центра кадра до центра маркера
            distance = np.linalg.norm(center - frame_center)
            # print(f"Расстояние от центра кадра до маркера {ids[i][0]}: {distance:.2f} пикселей")
            print(frame.shape[1], frame.shape[0])

    # Показываем изображение с маркерами
    cv2.imshow("Frame", frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрытие камеры и окон
cap.release()
cv2.destroyAllWindows()
