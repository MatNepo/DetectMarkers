import os

import cv2
import cv2.aruco as aruco

from detectors.aruco.params import custom_aruco_params
from detectors.aruco.generate import load_custom_aruco_dict

# Путь к видеофайлу
videos_folder = 'clean_cameras'
video_name = 'closet_video_1106_171241'
video_path = f'./{videos_folder}/{video_name}.avi'
custom_dict_path = "data/custom_dicts/DICT_5X5_10000.yml"

# Сохранение результатов
output_folder = './accuracy_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file_path = os.path.join(output_folder, f'{video_name}_accuracy.txt')

# Словарь маркеров для каждой камеры
# Пример структуры: замените camera_X на свои реальные имена камер/локаций
marker_ids_by_camera = {
    'camera_1': [0, 1, 2, 3],
    'camera_2': [10, 11, 12],
    'camera_3': [20, 21],
    # Добавьте свои камеры/локации по аналогии
}

# Определяем камеру на основе названия файла
camera = None
for camera_name in marker_ids_by_camera:
    if camera_name in video_name:
        camera = camera_name
        break

if camera is None:
    print("Не удалось определить камеру по названию видео.")
    exit()

# Получаем маркеры для выбранной камеры
marker_ids = marker_ids_by_camera[camera]

# Загрузка видео
cap = cv2.VideoCapture(video_path)

# Получаем общее количество кадров в видео
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Счётчики для каждого маркера
detected_frames_per_marker = {marker_id: 0 for marker_id in marker_ids}

aruco_dict = load_custom_aruco_dict(custom_dict_path)
aruco_params = custom_aruco_params(print_default_values=False, print_custom_values=False)

# Чтение кадров видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция маркеров на кадре
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected_candidates = detector.detectMarkers(frame)

    # Если маркеры найдены, обновляем счётчики
    if ids is not None:
        for marker_id in ids.flatten():
            if marker_id in marker_ids:
                detected_frames_per_marker[marker_id] += 1

# Закрываем видеофайл
cap.release()

# Рассчитываем процент времени, когда маркеры были обнаружены для каждого маркера
marker_accuracies = {}
for marker_id in marker_ids:
    accuracy = (detected_frames_per_marker[marker_id] / total_frames) * 100
    marker_accuracies[marker_id] = accuracy

# Рассчитываем общую точность
overall_accuracy = sum(marker_accuracies.values()) / len(marker_accuracies)

# Сохраняем результаты в файл
with open(output_file_path, 'w') as f:
    f.write(f"Точность для видео: {video_name}\n")
    for marker_id, accuracy in marker_accuracies.items():
        f.write(f"Маркер {marker_id}: {accuracy:.2f}%\n")
    f.write(f"\nОбщая точность для всех маркеров: {overall_accuracy:.2f}%\n")

print(f"Результаты сохранены в файл: {output_file_path}")
