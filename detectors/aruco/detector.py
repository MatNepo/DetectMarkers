import cv2
import numpy as np

from detectors.base.base_detector import BaseDetector


class ArucoDetector(BaseDetector):
    def __init__(self, camera_matrix, distortion_coefficients, target_ids, marker_metrics, total_frames,
                 custom_dict_path):
        super().__init__(camera_matrix, distortion_coefficients, target_ids, marker_metrics, total_frames,
                         custom_dict_path)
        self.custom_dict = self.load_custom_dict(custom_dict_path)

    def load_custom_dict(self, custom_dict_path):
        if not custom_dict_path:
            raise ValueError("Custom dict path cannot be empty")

        fs = cv2.FileStorage(custom_dict_path, cv2.FILE_STORAGE_READ)
        custom_dict = cv2.aruco.Dictionary()
        custom_dict.readDictionary(fs.root())
        fs.release()
        # print(f"Custom dictionary loaded from {custom_dict_path}.")
        return custom_dict

    def detect_markers(self, frame):
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.minMarkerDistanceRate = 0.02
        arucoParams.minMarkerPerimeterRate = 0.02  # 2
        arucoParams.minCornerDistanceRate = 0.02
        # arucoParams.useAruco3Detection = True
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        arucoParams.cornerRefinementWinSize = 3
        arucoParams.cornerRefinementMinAccuracy = 0.015
        arucoParams.relativeCornerRefinmentWinSize = 0.1

        detector = cv2.aruco.ArucoDetector(self.custom_dict, arucoParams)
        corners, ids, rejected_candidates = detector.detectMarkers(frame)
        return corners, ids

    def draw_markers(self, frame, corners, ids):
        return self.draw_aruco(frame, corners, ids)

    def draw_aruco(self, frame, corners, ids):
        text_position = (int(corners[0][0]), int(corners[0][1] - 10))
        cv2.putText(frame, f"ArUco: ID {ids}", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 0, 0))
        return frame
