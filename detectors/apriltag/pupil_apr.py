import cv2
from pupil_apriltags import Detector

at_detector = Detector(
    families="tagStandard52h13",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

video = cv2.VideoCapture("data/videos/example_video.mp4")

# Параметры для записи видео
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
output = cv2.VideoWriter(
    "output_with_tags_big.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray_frame)

    for tag in tags:
        print(f'tag: {tag}')
        corners = tag.corners.reshape(-1, 2)
        for i in range(4):
            start_point = tuple(corners[i].astype(int))
            end_point = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        center = tuple(tag.center.astype(int))
        cv2.putText(
            frame,
            str(tag.tag_id),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    if frame is not None:
        show_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', show_frame)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            video.release()
            cv2.destroyAllWindows()


    output.write(frame)



video.release()
output.release()
