import cv2
import os

import numpy as np
import pandas as pd

from detectors.aruco.ids_description import marker_ids_by_camera


def load_custom_aruco_dict(filename="data/custom_dicts/DICT_5X5_10000.yml"):
    """
    Load aruco dictionary from file

    :param filename: file with aruco dictionary, usually .yml file in './custom_dict/' folder
    :return: variable to work with custom aruco dictionary
    """
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    custom_dict = cv2.aruco.Dictionary()
    custom_dict.readDictionary(fs.root())
    fs.release()
    print(f"Custom dictionary loaded from {filename}.")
    return custom_dict


def object_points_def(marker_length):
    # Координаты углов метки в 3D (для solvePnP)
    half_marker_length = marker_length / 2
    object_points = np.array([
        [-half_marker_length, -half_marker_length, 0],
        [half_marker_length, -half_marker_length, 0],
        [half_marker_length, half_marker_length, 0],
        [-half_marker_length, half_marker_length, 0]
    ], dtype=np.float32)
    return object_points


def marker_square(coords):
    n = len(coords)
    if n != 4:
        return coords

    area = 0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    area_final = abs(area) / 2

    return area_final


def marker_square_2(corners):
    if corners.shape == (4, 2):
        return cv2.contourArea(corners)
    else:
        return corners


# Функция для обнаружения меток ArUco, подсчета расстояния, освещенности и точности
def detect_aruco_with_metrics(frame, custom_dict, camera_matrix, dist_coeffs, target_ids, marker_metrics, total_frames):
    # Настройка словаря ArUco и параметров
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.minMarkerDistanceRate = 0.02
    arucoParams.minMarkerPerimeterRate = 0.02  # 2
    arucoParams.minCornerDistanceRate = 0.02
    # arucoParams.useAruco3Detection = True
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    arucoParams.cornerRefinementWinSize = 3
    arucoParams.cornerRefinementMinAccuracy = 0.015
    arucoParams.relativeCornerRefinmentWinSize = 0.1

    detector = cv2.aruco.ArucoDetector(custom_dict, arucoParams)
    corners, ids, rejected_candidates = detector.detectMarkers(frame)

    metrics = []  # Список для хранения метрик для каждого кадра

    if ids is not None:
        ids = ids.flatten()
        print(f'ids: {ids}')
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        processed_ids = set()  # Для отслеживания обработанных меток в кадре

        for i, corner in enumerate(corners):
            marker_id = ids[i]
            #print(f'marker_id: {marker_id}')

            if marker_id in processed_ids or marker_id not in target_ids:
                continue  # Пропускаем метки, не входящие в категорию
            processed_ids.add(marker_id)
            # Получаем координаты углов маркера
            top_left = corner[0][0]  # (x1, y1)
            top_right = corner[0][1]  # (x2, y2)
            bottom_right = corner[0][2]  # (x3, y3)
            bottom_left = corner[0][3]  # (x4, y4)
            coordinates = [top_left, top_right, bottom_right, bottom_left]

            square = marker_square(coordinates)

            # для A4
            marker_length = 0.05  # Замените на реальный размер метки
            object_points = object_points_def(marker_length)
            # Используем cv2.solvePnP для оценки позы каждой метки
            retval, rvec, tvec = cv2.solvePnP(object_points, corner[0], camera_matrix, dist_coeffs)
            if retval:
                # Расстояние до метки в метрах
                distance = np.linalg.norm(tvec)

                # освещенность
                mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillPoly(mask, [corner[0].astype(int)], 255)
                # in bgr
                # mean_ligthness = np.mean(cv2.mean(frame, mask=mask)[:3])
                # mean_ligthness /= 255 # normalized
                # in Lab
                lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                mean_lightness = cv2.mean(lab_frame[:, :, 0], mask=mask)[0] / 255
                mean_lightness = mean_lightness

                # углы
                angles = []
                # Преобразуем rvec в матрицу вращения
                R, _ = cv2.Rodrigues(rvec)

                # Вычисляем углы Эйлера (в градусах)
                theta_x = abs(np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi )
                theta_y = abs(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)) * 180 / np.pi )
                theta_z = abs(np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi)
                angles.append((theta_x, theta_y, theta_z))
                # if marker_id==71:
                #     print(f'{marker_id}: square: {square}\n\t rvec:{rvec} theta_x: {theta_x} theta_y: {theta_y} theta_z: {theta_z}')
                ## accuracy
                detection_frames = {frame_num for frame_num, _ in marker_metrics[marker_id]["distance"]}
                detection_count = len(detection_frames)
                accuracy = detection_count / total_frames if total_frames > 0 else 0

                metrics.append((marker_id, distance, square, angles, mean_lightness, accuracy))

    return frame, metrics


def process_video(video_path, output_csv, custom_dict_path, targets, period_seconds=300):
    video_category = os.path.basename(os.path.dirname(video_path))  # video category (far, close, coffee etc)
    # marker_dict = targets.get(video_category, {})
    # category_ids = marker_dict.keys()
    category_ids = targets.get(video_category, [])
    if not category_ids:
        print(f"No target ids found for {video_category}")
        return

    custom_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    if custom_dict_path:
        custom_dict = load_custom_aruco_dict(custom_dict_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('no video')
        return

    camera_matrix = np.array([
        [4600, 0, cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2],
        [0, 4600, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2],
        [0, 0, 1]
    ], dtype=float)

    print(f'width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
    print(f'heigth: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')

    dist_coeffs = np.array([0.1, -0.1, 0.0, 0.0, 0.0])

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_period = int(fps * period_seconds)
    frame_count = 0

    # Счётчики обнаружений маркеров за период
    marker_metrics = {marker_id: {"distance": [], "square": [], "angles": [], "lightness": [], "accuracy": []}
                      for marker_id in category_ids}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Общее число кадров, просмотренных в текущем периоде
        frame_count += 1
        # здесь нужно продумать, что и как возвращать для каждого фрейма
        frame, metrics = detect_aruco_with_metrics(frame, custom_dict,
                                                   camera_matrix, dist_coeffs, category_ids,
                                                   marker_metrics, frame_count)

        for marker_id, distance, square, angles, lightness, accuracy in metrics:
            if marker_id in marker_metrics:
                frame_num = frame_count  # Номер текущего кадра
                existing_frames = {frame for frame, _ in marker_metrics[marker_id]["distance"]}

                # Добавляем только если текущий кадр ещё не обработан
                if frame_num not in existing_frames:
                    marker_metrics[marker_id]["distance"].append((frame_num, distance))
                    marker_metrics[marker_id]["square"].append((frame_num, square))
                    marker_metrics[marker_id]["angles"].append((frame_num, angles))
                    marker_metrics[marker_id]["lightness"].append((frame_num, lightness))
                    marker_metrics[marker_id]["accuracy"].append((frame_num, accuracy))

        if frame_count >= frames_per_period:
            print(f"frame_count: {frame_count} | frames_per_period: {frames_per_period}")
            print(f"Reached processing period, stopping")
            break

    cap.release()
    df = pd.DataFrame(columns=["period",
                               "video_category",
                               "marker_id",
                               "distance",
                               "square",
                               "angles",
                               "lightness",
                               "accuracy"])

    for marker_id, metrics in marker_metrics.items():
        if metrics['distance']:
            avg_distance = round(np.mean([d for _, d in metrics['distance']]), 2)
            avg_square = round(np.mean(metrics['square']), 2)
            avg_lightness = round(np.mean([l for _, l in metrics['lightness']]), 2)
            avg_accuracy = round(np.mean([acc for _, acc in metrics['accuracy']]), 2)
            if marker_id==71:
                print(f'marker_id: {marker_id}, angles: {metrics["angles"]}')
            avg_angles_class = np.round(np.mean([angle for _, angle in metrics['angles']], axis=0), 2)

            df = pd.concat([df, pd.DataFrame([{
                "period": period_seconds,
                "video_category": video_category,
                "marker_id": marker_id,
                "distance": avg_distance,
                "square": avg_square,
                "angles": avg_angles_class,
                "lightness": avg_lightness,
                "accuracy": avg_accuracy
            }])])

    #    cv2.destroyAllWindows()
    df.to_csv(output_csv, index=False)
    print(f'Saved data to {output_csv}')


if __name__ == '__main__':

    custom_dict_path = "data/custom_dicts/DICT_5X5_10000.yml"
    video_path = "data/videos/example_video.mp4"
    output_fol = "reports_final_ok_nata"
    if not os.path.exists(output_fol):
        os.makedirs(output_fol)
    csv_name = (os.path.basename(os.path.dirname(video_path)) + "_" +
                os.path.splitext(os.path.basename(video_path))[0]) + "_report.csv"

    output_csv = os.path.join(output_fol, csv_name)
    target_ids = marker_ids_by_camera
    print(f'target_ids: {target_ids}')
    process_video(video_path=video_path,
                  output_csv=output_csv,
                  custom_dict_path=custom_dict_path,
                  targets=target_ids,
                  period_seconds=60)
    '''
        1. open video for watch
        2. detect marker
        3. calculate distance, angle, lightness - dur some time
        4. calc acc - dur some time
        5. mean all the values
        6. write down to csv
        7.
        видео поместить в отдельную папку и именовать посл каталог по категориям
    '''
