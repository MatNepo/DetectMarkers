import sys
import time

import cv2
import os

import numpy as np
import pandas as pd

from aruco.aruco_detector import ArucoDetector
from aruco.aruco_ids_description import aruco_ids_by_camera

from detectors.stag.detector import StagDetector
from detectors.stag.ids_description import stag_ids_by_camera

from detectors.apriltag.detector import AprilTagDetector
from detectors.apriltag.ids_description import apriltag_ids_by_camera

import config


def show_progress_bar(total_frames, current_frame):
    bar_length = 50  # Длина прогресс-бара

    progress = current_frame / total_frames
    block = int(round(bar_length * progress))
    progress_bar = f"Progress: [{'#' * block}{'.' * (bar_length - block)}] {current_frame:.2f}/{total_frames:.2f} frames"
    sys.stdout.write(f"\r{progress_bar}")
    sys.stdout.flush()

    time.sleep(0.1)  # Обновляем прогресс каждые 100 мс


# Функция для обнаружения маркеров
def detect_markers_with_metrics(frame, camera_matrix, dist_coeffs, target_ids, marker_metrics, total_frames,
                                marker_type):
    metrics = []

    if marker_type == 'aruco':
        aruco_dict_path = config.custom_dict_path_aruco
        detector = ArucoDetector(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            target_ids=target_ids,
            marker_metrics=marker_metrics,
            total_frames=total_frames,
            custom_dict_path=aruco_dict_path,
        )
    elif marker_type == 'stag':
        stag_dict_path = config.custom_dict_path_stag
        detector = StagDetector(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            target_ids=target_ids,
            marker_metrics=marker_metrics,
            total_frames=total_frames,
            dictionary_indices=stag_dict_path)
    elif marker_type == 'apriltag':
        apriltag_dict_path = config.custom_dict_path_apriltag
        detector = AprilTagDetector(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            target_ids=target_ids,
            marker_metrics=marker_metrics,
            total_frames=total_frames,
            family=apriltag_dict_path
        )
    else:
        raise ValueError(f'marker_type {marker_type} not supported')
    print(f'marker_type: {marker_type}')
    try:
        frame, marker_metrics = detector(frame)
    except Exception as exc:
        print(f'Exception {exc} on marker {marker_type}')
    return frame, metrics


# Функция для отображения маркеров
def show_markers(camera_matrix, dist_coeffs, frame, target_ids, marker_metrics, total_frames, marker_type):
    if marker_type == 'aruco':
        aruco_dict_path = config.custom_dict_path_aruco
        detector = ArucoDetector(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            target_ids=target_ids,
            marker_metrics=marker_metrics,
            total_frames=total_frames,
            custom_dict_path=aruco_dict_path,
        )
    elif marker_type == 'stag':
        stag_dict_path = config.custom_dict_path_stag
        detector = StagDetector(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            target_ids=target_ids,
            marker_metrics=marker_metrics,
            total_frames=total_frames,
            dictionary_indices=stag_dict_path
        )
    elif marker_type == 'apriltag':
        apriltag_dict_path = config.custom_dict_path_apriltag
        detector = AprilTagDetector(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            target_ids=target_ids,
            marker_metrics=marker_metrics,
            total_frames=total_frames,
            family=apriltag_dict_path
        )
    else:
        raise ValueError(f'marker_type {marker_type} not supported')
    # print(f'marker_type: {marker_type}')
    try:
        frame = detector.show_detected_markers(frame)
    except Exception as exc:
        print(f'Exception {exc} on marker {marker_type}')
    return frame


def save_video(video_path, output_video, targets, period_seconds=300, types_for_detection=None):
    video_category = os.path.basename(os.path.dirname(video_path))  # video category (far, close, coffee etc)
    category_ids = {
        marker_type: targets.get(marker_type, {}).get(video_category, [])
        for marker_type in types_for_detection
    }
    if not any(category_ids.values()):
        print(f"No target ids found for {video_category}")
        return

    print(f'video_path: {video_path}')
    print(f'category_ids: {category_ids}')

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('no video')
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f'Video resolution: {width}x{height}')

    camera_matrix = np.array([
        [4600, 0, cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2],
        [0, 4600, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2],
        [0, 0, 1]
    ], dtype=float)

    dist_coeffs = np.array([0.1, -0.1, 0.0, 0.0, 0.0])

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (int(width), int(height)))

    frames_per_period = int(fps * period_seconds)
    frame_count = 0

    # Счётчики обнаружений маркеров за период
    marker_metrics = {
        marker_type: {
            marker_id: {"distance": [], "square": [], "angles": [], "lightness": [], "accuracy": []}
            for marker_id in ids
        }
        for marker_type, ids in category_ids.items()
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Общее число кадров, просмотренных в текущем периоде
        frame_count += 1
        show_progress_bar(total_frames=frames_per_period, current_frame=frame_count)
        for marker_type in types_for_detection:
            # print(f'num frame: {frame_count}')
            if not category_ids[marker_type]:
                continue
            # сделаем копию фрейма,чтобы результаты обработки  пред маркера при отрисовке не передавались для др маркера
            # frame_copy = frame.copy()
            try:
                frame = show_markers(camera_matrix,
                                     dist_coeffs,
                                     frame,
                                     category_ids[marker_type],
                                     marker_metrics[marker_type],
                                     frame_count,
                                     marker_type)
            except Exception as exc:
                print(f'Exception {exc} on {frame_count} with detection {marker_type} marker!')

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(frame)

        if frame_count >= frames_per_period:
            print(f"frame_count: {frame_count} | frames_per_period: {frames_per_period}")
            print(f"Reached processing period, stopping")
            break

    print(f'Saved all data to {output_video}')

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path, output_csv, targets, period_seconds=300, types_for_detection=None):
    video_category = os.path.basename(os.path.dirname(video_path))  # video category (far, close, coffee etc)
    category_ids = {
        marker_type: targets.get(marker_type, {}).get(video_category, [])
        for marker_type in types_for_detection
    }
    if not any(category_ids.values()):
        print(f"No target ids found for {video_category}")
        return

    print(f'video_path: {video_path}')
    print(f'category_ids: {category_ids}')

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
    marker_metrics = {
        marker_type: {
            marker_id: {"distance": [], "square": [], "angles": [], "lightness": [], "accuracy": []}
            for marker_id in ids
        }
        for marker_type, ids in category_ids.items()
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Общее число кадров, просмотренных в текущем периоде
        frame_count += 1
        for marker_type in types_for_detection:
            print(f'num frame: {frame_count}')
            if not category_ids[marker_type]:
                continue
            #сделаем копию фрейма,чтобы результаты обработки  пред маркера при отрисовке не передавались для др маркера
            #frame_copy = frame.copy()
            try:
                frame, metrics = detect_markers_with_metrics(frame,
                                                             camera_matrix, dist_coeffs, category_ids[marker_type],
                                                             marker_metrics[marker_type], frame_count, marker_type)

                for marker_id, distance, square, angles, lightness, accuracy in metrics:
                    if marker_id in marker_metrics[marker_type]:
                        frame_num = frame_count  # Номер текущего кадра
                        existing_frames = {frame for frame, _ in marker_metrics[marker_type][marker_id]["distance"]}

                        # Добавляем только если текущий кадр ещё не обработан
                        if frame_num not in existing_frames:
                            marker_metrics[marker_type][marker_id]["distance"].append((frame_num, distance))
                            marker_metrics[marker_type][marker_id]["square"].append((frame_num, square))
                            marker_metrics[marker_type][marker_id]["angles"].append((frame_num, angles))
                            marker_metrics[marker_type][marker_id]["lightness"].append((frame_num, lightness))
                            marker_metrics[marker_type][marker_id]["accuracy"].append((frame_num, accuracy))
            except Exception as exc:
                print(f'Exception {exc} on {frame_count} with detection {marker_type} marker!')

        if frame_count >= frames_per_period:
            print(f"frame_count: {frame_count} | frames_per_period: {frames_per_period}")
            print(f"Reached processing period, stopping")
            break

    cap.release()

    # Создаем отдельные DataFrame для каждого типа маркеров
    dataframes = {marker_type: pd.DataFrame() for marker_type in types_for_detection}
    # для общего датафрейма
    combined_dataframes = []
    for marker_type, type_metrics in marker_metrics.items():
        rows = []
        for marker_id, metrics in type_metrics.items():
            if metrics['distance']:
                avg_distance = round(np.mean([d for _, d in metrics['distance']]), 2)
                avg_square = round(np.mean(metrics['square']), 2)
                avg_lightness = round(np.mean([l for _, l in metrics['lightness']]), 2)
                avg_accuracy = round(np.mean([acc for _, acc in metrics['accuracy']]), 2)
                avg_angles_class = np.round(np.mean([angle for _, angle in metrics['angles']], axis=0), 2)

                rows.append({
                    "marker_id": marker_id,
                    "distance": avg_distance,
                    "square": avg_square,
                    "angles": avg_angles_class,
                    "lightness": avg_lightness,
                    "accuracy": avg_accuracy,
                    "video_category": video_category
                })

        df = pd.DataFrame(rows)
        df['marker_type'] = marker_type
        new_order = ['marker_type'] + [col for col in df.columns if col != 'marker_type']
        df = df[new_order]
        combined_dataframes.append(df)
        # Сохраняем каждый DataFrame в отдельный файл
        type_output_csv = output_csv.replace(".csv", f"_{marker_type}.csv")
        df.to_csv(type_output_csv, index=False)
        print(f"Saved data for {marker_type} to {type_output_csv}")

    combined_dataframes = pd.concat(combined_dataframes, ignore_index=True)
    combined_dataframes.to_csv(output_csv, index=False)
    print(f'Saved all data to {output_csv}')

    cv2.destroyAllWindows()


if __name__ == '__main__':

    video_path = './videos/office_table/video_2024-11-25_14-30-54.mp4'
    # os.add_dll_directory("C:/Users/mnepomnyaschiy/miniconda3/envs/all_tags/Lib/site-packages/pupil_apriltags.libs")
    # Для работы с pupil_apriltags укажите путь к DLL через os.add_dll_directory(), если требуется

    output_fol = "reports_3types_office_table"
    if not os.path.exists(output_fol):
        os.makedirs(output_fol)
    csv_name = (os.path.basename(os.path.dirname(video_path)) + "_" +
                os.path.splitext(os.path.basename(video_path))[0]) + "report.csv"
    video_name = (os.path.basename(os.path.dirname(video_path)) + "_" +
                  os.path.splitext(os.path.basename(video_path))[0]) + "video.avi"

    output_csv = os.path.join(output_fol, csv_name)
    output_video = os.path.join(output_fol, video_name)

    target_ids = {'aruco': aruco_ids_by_camera,
                  'stag': stag_ids_by_camera,
                  'apriltag': apriltag_ids_by_camera,
                  }
    print(len(target_ids))
    print(f'target_ids: {target_ids}')

    save_video_flag = True

    if save_video_flag:
        save_video(video_path=video_path,
                   output_video=output_video,
                   targets=target_ids,
                   period_seconds=5,
                   # types_for_detection=['aruco', 'stag', 'apriltag'],
                   types_for_detection=['stag'],
                   )
    else:
        process_video(video_path=video_path,
                      output_csv=output_csv,
                      targets=target_ids,
                      period_seconds=5,
                      types_for_detection=['aruco', 'stag', 'apriltag']
                      )  # 'aruco', 'stag','apriltag'

'''
последняя папка в кот лежит видос-должна быть такого же названия как имя семейства в описании каждой метки (пример 'close','office_table')
в файле config.py  для аруко указан путь к словарю, для стаг-номера словарей, для эприлтэг-имя(одно!) словаря
'''
