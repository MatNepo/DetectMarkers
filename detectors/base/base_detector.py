from abc import ABC, abstractmethod
import cv2
import numpy as np


class BaseDetector:
    def __init__(self, camera_matrix, distortion_coefficients, target_ids, marker_metrics, total_frames,
                 custom_dict_path=None):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.target_ids = target_ids or set()
        self.total_frames = total_frames
        self.marker_metrics = marker_metrics
        self.custom_dict = None

    def __call__(self, frame):
        corners, ids = self.detect_markers(frame)  # Вызываем детекцию маркеров (метод должен быть переопределён)
        return self.process_frame(frame, corners, ids)

    def show_detected_markers(self, frame):
        corners, ids = self.detect_markers(frame)
        return self.draw_markers(frame, corners, ids)

    def detect_markers(self, frame):
        """
        Метод для детектирования маркеров.
        Должен быть переопределён в дочерних классах.
        """
        raise NotImplementedError("Этот метод должен быть реализован в подклассе.")

    def process_frame(self, frame, corners, ids):
        metrics = []  # Список для хранения метрик для каждого кадра

        if ids is not None:
            # print(f'ids: {ids}')
            if isinstance(ids, list):
                # convert to numpy array
                ids = np.array(ids)

            ids = ids.flatten()
            processed_ids = set()  # Для отслеживания обработанных меток в кадре
            for i, corner in enumerate(corners):
                marker_id = ids[i]
                if marker_id in processed_ids or marker_id not in self.target_ids:
                    continue  # Пропускаем метки, не входящие в категорию
                processed_ids.add(marker_id)

                # площадь
                square = self._count_square(corner)
                # углы и расстояние
                distance, angles = self._calculate_pose(self, corner)

                # освещенность
                mean_lightness = self._calculate_lightness(frame, corner)

                # accuracy
                accuracy = self._calculate_accuracy(marker_id)

                metrics.append((marker_id, distance, square, angles, mean_lightness, accuracy))

            #frame = self.draw_markers(frame, corners, ids) # включим когда реализуем сохранение готового видоса

        return frame, metrics

    @staticmethod
    def _count_square(corner):
        # print(f'corner_0 {corner}')
        # if corner.ndim == 3: # like aruco and stag
        #     print('aruco|stag', type(corner[0]))
        #     contour = corner[0]
        # elif corner.ndim == 2:
        #     print('apriltag ', type(corner))
        #     contour = np.expand_dims(corner, axis=0)
        #     print(f'corner_2 {contour}')
        # else:
        #    raise ValueError('Incorrect number of dimensions of corners')

        # попробуем эту
        return cv2.contourArea(corner[0])  #contour

    @staticmethod
    def _define_object_points(marker_length):
        """
        Определение 3D-координат углов маркера.
        """
        # Координаты углов метки в 3D (для solvePnP)
        half_marker_length = marker_length / 2
        return np.array([
            [-half_marker_length, -half_marker_length, 0],
            [half_marker_length, -half_marker_length, 0],
            [half_marker_length, half_marker_length, 0],
            [-half_marker_length, half_marker_length, 0]
        ], dtype=np.float32)

    @staticmethod
    def _calculate_pose(self, corner):
        marker_length = 0.05  # Реальный размер маркера
        object_points = self._define_object_points(marker_length)
        # Используем cv2.solvePnP для оценки позы каждой метки
        retval, rvec, tvec = cv2.solvePnP(object_points, corner[0], self.camera_matrix, self.distortion_coefficients)
        distance = np.linalg.norm(tvec) if retval else None
        angles = self._calculate_angles(rvec) if retval else None
        return distance, angles

    @staticmethod
    def _calculate_angles(rvec):
        """
        Вычисление углов ориентации (Эйлера) на основе вектора вращения.
        """
        # Преобразуем rvec в матрицу вращения
        R, _ = cv2.Rodrigues(rvec)
        # Вычисляем углы Эйлера (в градусах)
        theta_x = abs(np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi)
        theta_y = abs(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)) * 180 / np.pi)
        theta_z = abs(np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi)
        return theta_x, theta_y, theta_z

    def _calculate_lightness(self, frame, corner):
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [corner[0].astype(int)], 255)
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mean_lightness = cv2.mean(lab_frame[:, :, 0], mask=mask)[0] / 255
        return mean_lightness

    def _calculate_accuracy(self, marker_id):
        detection_frames = {frame_num for frame_num, _ in self.marker_metrics[marker_id]["distance"]}
        detection_count = len(detection_frames)
        accuracy = detection_count / self.total_frames if self.total_frames > 0 else 0
        return accuracy

    def __getitem__(self, item):
        return self.marker_metrics[item]

    def draw_markers(self, frame, corners, ids):
        raise NotImplementedError('Method must be implemented in derived class.')
