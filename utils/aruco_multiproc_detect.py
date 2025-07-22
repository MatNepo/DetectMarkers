from datetime import datetime
import multiprocessing as mp
import os

import cv2
import numpy as np


def write(frame: np.ndarray, out_folder: str, stream: str) -> str:
    date_time = datetime.now()
    date = f"{date_time.year}{date_time.month:02}{date_time.day:02}"
    time = f"{date_time.hour:02}{date_time.minute:02}{date_time.second:02}"
    date_folder = os.path.join(out_folder, stream, date)
    if not os.path.isdir(date_folder):
        os.makedirs(date_folder)
    filename = f"{stream}_{date}_{time}.jpg"
    path = os.path.join(date_folder, filename)
    ret = cv2.imwrite(path, frame)
    if ret:
        print(path, "Written!")
    else:
        print("cannot write")
    return path


def detect_aruco(frame, is_debug=True):
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    # aruco detector parameters
    arucoParams = cv2.aruco.DetectorParameters()

    arucoParams.minMarkerDistanceRate = 0.02
    arucoParams.minMarkerPerimeterRate = 0.02
    arucoParams.minCornerDistanceRate = 0.02
    # arucoParams.useAruco3Detection = True
    # arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    arucoParams.cornerRefinementWinSize = 3
    arucoParams.cornerRefinementMinAccuracy = 0.025
    # arucoParams.relativeCornerRefinmentWinSize = 0.01
    # arucoParams.polygonalApproxAccuracyRate = 0.1
    # arucoParams.adaptiveThreshWinSizeMin = 1
    # arucoParams.adaptiveThreshWinSizeMin = 50
    # arucoParams.adaptiveThreshWinSizeStep = 3
    # arucoParams.polygonalApproxAccuracyRate = 0.5
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejectedCandidates = detector.detectMarkers(frame)

    if len(corners) > 0:
        ids = ids.flatten()

        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners, ids):
            # Extract the marker corners
            corners = marker_corner.reshape((4, 2))

            (top_left, top_right, bottom_right, bottom_left) = corners

            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))

            # Draw the bounding box of the ArUco detection
            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

            # Calculate and draw the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            # Draw the ArUco marker ID on the video frame
            # The ID is always located at the top_left of the ArUco marker
            cv2.putText(frame, str(marker_id),
                        (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
    if is_debug:
        if len(rejectedCandidates) > 0:
            for rej_cand in rejectedCandidates:
                rej_cand_corn = rej_cand.reshape((4, 2)).astype(int)
                (tl, tr, br, bl) = rej_cand_corn
                cv2.line(frame, tl, tr, (0, 0, 255), 2)
                cv2.line(frame, tr, br, (0, 0, 255), 2)
                cv2.line(frame, br, bl, (0, 0, 255), 2)
                cv2.line(frame, bl, tl, (0, 0, 255), 2)
                center_x = int((tl[0] + br[0]) / 2.0)
                center_y = int((tl[1] + br[1]) / 2.0)
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 255), -1)

    return frame


def clahe_img(img_bgr, clahe_obj):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = clahe_obj.apply(img_hsv[:, :, 2])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def watch(path, is_write=False):
    ip = path.split("=")[-1]
    cap = cv2.VideoCapture(path)
    if is_write:
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        out = cv2.VideoWriter(f"aruco/rotation.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (int(w), int(h)))
    if not cap.isOpened():
        print('no video')
        cv2.destroyAllWindows()
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = detect_aruco(frame)
                show_frame = cv2.resize(frame, (1920, 1080))
                cv2.imshow(ip, show_frame)
                k = cv2.waitKey(1)
                if is_write:
                    out.write(frame)
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    break
                if k == ord('s'):
                    write(frame, "aruco", ip)
            else:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":

    streams = [
        "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_1",
        "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_2",
        "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_3",
        "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_4",
        "rtsp://user:password@host:554/rtsp?channelid=CAMERA_ID_5",
    ]

    small = False
    if small:
        streams = [i + "&streamtype=alternative" for i in streams]

    processes = []
    for num, name in enumerate(streams):
        process = mp.Process(target=watch, args=(name, False))
        processes.append(process)
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
