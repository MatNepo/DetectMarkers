import cv2
import os
import numpy as np
import stag
def process_video(video_path, period_seconds=300):

    print(f'video_path: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('no video')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_period = int(fps * period_seconds)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # detect markers
        corners, ids, rejected_corners = stag.detectMarkers(frame, 23) # 11
        print(f'ids: {ids}')
        if ids is not None:
            # draw detected markers with ids
            stag.drawDetectedMarkers(frame, corners, ids)
            # draw rejected quads without ids with different color
            stag.drawDetectedMarkers(frame, rejected_corners, border_color=(255, 0, 0))

        if frame is not None:
            show_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('frame', show_frame)

            k = cv2.waitKey(1)&0xff
            if k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return


        # Общее число кадров, просмотренных в текущем периоде
        frame_count += 1


        if frame_count >= frames_per_period:
            print(f"frame_count: {frame_count} | frames_per_period: {frames_per_period}")
            print(f"Reached processing period, stopping")
            break

    cap.release()



if __name__ == '__main__':

    video_path = 'data/videos/example_video.avi'

    process_video(video_path=video_path,
                  period_seconds=30,
)