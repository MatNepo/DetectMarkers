from array import array

import cv2
import numpy as np
import stag

from detectors.base.base_detector import BaseDetector

class StagDetector(BaseDetector):
    def __init__(self, camera_matrix, distortion_coefficients, target_ids, marker_metrics, total_frames, dictionary_indices):
        super().__init__(camera_matrix, distortion_coefficients, target_ids, marker_metrics, total_frames)
        self.dictionary_indices = dictionary_indices


    def detect_markers(self, frame):
        all_corners = []
        all_ids = []
        for index in self.dictionary_indices:
            corners, ids, rejected_corners = stag.detectMarkers(frame, index, errorCorrection=-1)
            if ids is not None:
                all_corners.extend(corners)
                all_ids.extend(ids)
        return all_corners, all_ids

    def draw_markers(self, frame, corners, ids):
        return self.draw_stag(frame, corners, ids)

    def draw_stag(self, frame, corners, ids):
        # draw detected markers with ids
        all_corners = np.expand_dims(corners, axis=0).astype(np.float32)
        # print(all_corners)
        for i, corner in enumerate(all_corners):
            # print(ids)
            text_position = (int(corner[0][0][0]), int(corner[0][0][1] - 10))
            cv2.putText(frame, f"STag: ID {ids[i][0]}", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        stag.drawDetectedMarkers(frame, corners, ids, border_color=(0, 0, 255))
        return frame