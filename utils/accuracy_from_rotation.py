import csv
from collections import defaultdict

# �������������� ������� ��� �������� ������ �� ������� �������
detection_data = defaultdict(lambda: {"detections": 0, "attempts": 0, "tvec_error_sum": 0, "rvec_error_sum": 0})

def log_detection_data(marker_id, yaw, pitch, roll, success, tvec_error, rvec_error):
    data = detection_data[marker_id]
    data["attempts"] += 1
    if success:
        data["detections"] += 1
        data["tvec_error_sum"] += tvec_error
        data["rvec_error_sum"] += rvec_error

def save_results_to_csv(filename="data/detection_analysis.csv"):
    with open(filename, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Marker ID", "Yaw", "Pitch", "Roll", "Detection Rate", "Avg Tvec Error", "Avg Rvec Error"])
        for marker_id, data in detection_data.items():
            detection_rate = data["detections"] / data["attempts"] if data["attempts"] > 0 else 0
            avg_tvec_error = data["tvec_error_sum"] / data["detections"] if data["detections"] > 0 else 0
            avg_rvec_error = data["rvec_error_sum"] / data["detections"] if data["detections"] > 0 else 0
            writer.writerow([marker_id, yaw, pitch, roll, detection_rate, avg_tvec_error, avg_rvec_error])

            # � ����� ����������� ��������
            if retval:
                euler_angles = rotation_vector_to_euler(rvec)
                tvec_error = np.linalg.norm(tvec - true_tvec)
                rvec_error = np.linalg.norm(rvec - true_rvec)
                log_detection_data(marker_id, euler_angles[0], euler_angles[1], euler_angles[2], True, tvec_error, rvec_error)
            else:
                log_detection_data(marker_id, euler_angles[0], euler_angles[1], euler_angles[2], False, 0, 0)
