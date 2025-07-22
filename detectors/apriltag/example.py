import cv2
import numpy as np
# from apriltag import apriltag
# #import apriltag
#
# imagepath = 'apriltag-imgs-master/test/41h12/1.jpg'
# image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
# detector = apriltag("tagStandard41h12")
#
# detections = detector.detect(image)
from pupil_apriltags import Detector

imagepath = 'data/example_images/2.jpg'
frame = cv2.imread(imagepath)
print(frame.shape)
at_detector = Detector(
    families="tagStandard41h12",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

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
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

