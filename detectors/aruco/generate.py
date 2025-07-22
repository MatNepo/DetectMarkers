import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from save_dict import dict_filename, bit_size, unique_ids


def load_custom_aruco_dict(filename="data/custom_dicts/DICT_5X5_10000.yml"):
    # print('Start reading dict')
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    custom_dict = cv2.aruco.Dictionary()
    custom_dict.readDictionary(fs.root())
    fs.release()
    print(f"Custom dictionary loaded from {filename}.")
    return custom_dict


def generate(pix_size=1500,
             unique_ids=unique_ids,
             ids_num=unique_ids,
             bit_size=bit_size,
             low_contrast=0,
             dictionary=None,
             ):
    """

    :param dictionary: custom dictionary
    :param pix_size: marker size in pixels
    :param unique_ids: number of unique ids (50, 100, 250 or 1000)
    :param ids_num: number of markers to generate. If ids_num set to <= 0, than it equals 'unique_ids' param
    :param bit_size: marker size in bits (5, 6 or 7)
    :param low_contrast: if > 0, sets black bits to low_contrast value. Should be from 0 to 255
    :return:
    """
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        # "DICT_7X7_10000": cv2.aruco.Dictionary 00000228DF8C7DB0
    }

    rb = lb = (2100 - pix_size) // 2
    tb = bb = (2970 - pix_size) // 2

    if ids_num <= 0:
        ids_num = unique_ids

    tag_type = f"DICT_{bit_size}X{bit_size}_{unique_ids}"
    # dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[tag_type])

    if not os.path.exists(tag_type):
        os.makedirs(tag_type)
    for id_ in range(0, ids_num):
        print(id_)
        tag = np.zeros((pix_size, pix_size, 1), dtype="uint8")
        cv2.aruco.generateImageMarker(dictionary, id_, pix_size, tag, 1)
        border_tag = cv2.copyMakeBorder(tag, *(tb, bb, lb, rb), cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(border_tag, f"{bit_size}X{bit_size}_{unique_ids} id: {id_}", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 2,
                    (0, 0, 0), 2, cv2.LINE_AA)
        if low_contrast > 0:
            border_tag[border_tag == 0] = low_contrast
            write_path = os.path.join(tag_type, "low_contrast", f"{id_}.png")
        else:
            write_path = os.path.join(tag_type, f"{id_}.png")
        print(write_path)
        ret = cv2.imwrite(write_path, border_tag)
        print(ret)
        cv2.imshow(f"{id_}", border_tag)
        cv2.waitKey(1)
        cv2.destroyWindow(f"{id_}")
    return


if __name__ == "__main__":
    custom_dict = load_custom_aruco_dict()
    generate(dictionary=custom_dict)
