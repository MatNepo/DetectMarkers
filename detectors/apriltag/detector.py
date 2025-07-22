import cv2
import numpy as np
from pupil_apriltags import Detector

from detectors.base.base_detector import BaseDetector


class AprilTagDetector(BaseDetector):
    def __init__(self, camera_matrix, distortion_coefficients, target_ids, marker_metrics, total_frames,
                 family="tagStandard41h12"):
        super().__init__(camera_matrix, distortion_coefficients, target_ids, marker_metrics, total_frames)
        self.family = family
        self.at_detector = Detector(
            families=self.family,
            nthreads=4,  # 1
            # quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def detect_markers(self, frame):

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print('detecting apriltag')
        tags = self.at_detector.detect(gray_frame)
        # print('finish')
        all_corners = [np.expand_dims(tag.corners, axis=0).astype(np.float32) for tag in tags]
        all_ids = [tag.tag_id for tag in tags]
        # print('all_ids')
        return all_corners, all_ids

    def draw_markers(self, frame, corners, ids):
        return self.draw_apriltag(frame, corners, ids)

    def draw_apriltag(self, frame, corners, ids):
        # draw detected markers with ids
        for i, corner in enumerate(corners):
            marker_id = ids[i]
            corner_points = np.array(corner).squeeze(axis=0).astype(np.int32)
            center_x = int(np.mean(corner_points[:, 0]))
            center_y = int(np.mean(corner_points[:, 1]))

            for j in range(4):
                cv2.line(frame, tuple(corner_points[j]), tuple(corner_points[(j + 1) % 4]), (0, 255, 0), 2)
            text_position = (center_x, center_y - 15)
            cv2.putText(frame, f'AprilTag ID: {marker_id}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 216, 255),
                        2)
        return frame
