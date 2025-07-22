import cv2
import numpy as np
from pupil_apriltags import Detector

def get_stream_id(camera_name):
    # Пример структуры: замените camera_X на свои реальные имена камер/локаций
    cameras_ip_dict = {
        "camera_1": "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_1",
        "camera_2": "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_2",
        "camera_3": "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_3",
        # Добавьте свои камеры/локации по аналогии
    }
    return cameras_ip_dict[camera_name]


def return_marker_distance(camera_name):
    dist_coeffs = np.array([0.1, -0.1, 0.0, 0.0, 0.0])

    # Инициализация видео
    stream = get_stream_id(camera_name=camera_name)
    print(stream)
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        print("Ошибка: не удается открыть поток")
        return

    # Настройка детектора AprilTag
    at_detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

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
        if not ret:
            break

        # Преобразуем кадр в градации серого для детектирования меток
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray_frame)

        # Устанавливаем параметры камеры для solvePnP
        height, width = frame.shape[:2]
        camera_matrix = np.array([
            [4600, 0, width / 2],  # fx, cx
            [0, 4600, height / 2],  # fy, cy
            [0, 0, 1]
        ], dtype=float)

        # Обрабатываем каждую обнаруженную метку
        for tag in tags:
            # Углы метки и ее центр
            corners = tag.corners.reshape(-1, 2)
            marker_id = tag.tag_id

            # Определяем длину метки
            if marker_id == 20:  # для A3
                marker_length = 0.075
            else:  # для A4
                marker_length = 0.05  # замените на реальный размер метки

            # 3D координаты углов метки
            object_points = object_points_def(marker_length)

            # Оценка позиции метки с помощью solvePnP
            retval, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

            if retval:
                # Расстояние до метки
                distance = np.linalg.norm(tvec)
                text_position = (int(tag.center[0]), int(tag.center[1] - 10))
                cv2.putText(frame, f"ID {marker_id}: {distance:.2f} m", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Рисуем рамку вокруг метки
                for i in range(4):
                    start_point = tuple(corners[i].astype(int))
                    end_point = tuple(corners[(i + 1) % 4].astype(int))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Отображаем кадр с аннотациями
        resize_coef = 0.95
        new_width = int(width * resize_coef)
        new_height = int(height * resize_coef)
        show_frame_resized = cv2.resize(frame, (new_width, new_height))
        cv2.imshow(f"{camera_name}", show_frame_resized)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    camera = input("Enter camera name: ")
    return_marker_distance(camera)
