import cv2
import numpy as np
from detectors.aruco.params import custom_aruco_params


def load_custom_dict(filename="DICT_5X5_10000.yml"):
    """
    Load aruco dictionary from file

    :param filename: file with aruco dictionary, usually .yml file in './custom_dict/' folder
    :return: variable to work with custom aruco dictionary
    """
    # print('Start reading dict')
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    custom_dict = cv2.aruco.Dictionary()
    custom_dict.readDictionary(fs.root())
    fs.release()
    print(f"Custom dictionary loaded from {filename}.")
    return custom_dict


def get_stream_id(camera_name):
    # Пример структуры: замените camera_X на свои реальные имена камер/локаций
    cameras_ip_dict = {
        "camera_1": "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_1",
        "camera_2": "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_2",
        "camera_3": "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_3",
        # Добавьте свои камеры/локации по аналогии
    }
    return cameras_ip_dict[camera_name]


def return_marker_distance(camera_name, file_path="data/custom_dicts/DICT_5X5_10000.yml"):
    dist_coeffs = np.array([0.1, -0.1, 0.0, 0.0, 0.0])

    # Инициализация видео
    stream = get_stream_id(camera_name=camera_name)
    print(stream)
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        print("Ошибка: не удается открыть поток")

    # Задаем параметры для распознавания ArUco-меток
    aruco_dict = load_custom_dict(filename=file_path)
    parameters = custom_aruco_params(print_default_values=False, print_custom_values=False)

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

    while cap.isOpened():
        ret, frame = cap.read()
        height, width, channels = frame.shape
        # Значения для матрицы камеры и коэффициентов искажений
        camera_matrix = np.array([
            [4600, 0, width / 2],  # fx, cx
            [0, 4600, height / 2],  # fy, cy
            [0, 0, 1]
        ], dtype=float)
        if not ret:
            break

        # Распознаем ArUco-метки
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected_candidates = detector.detectMarkers(frame)

        if ids is not None:
            # Отображаем найденные метки
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Определяем расстояние до каждой метки
            for i, corner in enumerate(corners):

                if ids[i][0] == 20:  # для A3
                    marker_length = 0.075
                else:  # для A4
                    marker_length = 0.05  # Замените на реальный размер метки

                object_points = object_points_def(marker_length)

                # Используем cv2.solvePnP для оценки позы каждой метки
                retval, rvec, tvec = cv2.solvePnP(object_points, corner[0], camera_matrix, dist_coeffs)

                if retval:
                    # Расстояние до метки в метрах
                    distance = np.linalg.norm(tvec)

                    # Отображаем расстояние рядом с меткой
                    marker_id = ids[i][0]
                    text_position = (int(corner[0][0][0]), int(corner[0][0][1] - 10))
                    cv2.putText(frame, f"ID {marker_id}: {distance:.2f} m", text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Отображаем оси координат метки
                    # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 1.5)

        # Отображаем кадр с аннотациями
        resize_coef = 0.95
        new_width = int(1920 * resize_coef)
        new_height = int(1080 * resize_coef)
        show_frame_resized = cv2.resize(frame, (new_width, new_height))
        cv2.imshow(f"{camera}", show_frame_resized)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    camera = input("Enter camera name: ")
    file_path = "data/custom_dicts/DICT_5X5_10000.yml"
    return_marker_distance(camera, file_path)
