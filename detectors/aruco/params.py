import cv2


def custom_aruco_params(print_default_values=False, print_custom_values=False):
    """
        Function sets custom aruco parameters for aruco marker.

        :var int minMarkerDistanceRate: minimum distance between any pair of corners from two different markers.
        :var int minMarkerPerimeterRate: minimum size of a marker.
        :var int minCornerDistanceRate: minimum distance between any pair of corners from two different markers.
        :var int minOtsuStdDev: minimum standard deviation of the pixel values to perform Otsu thresholding.
        :var bool useAruco3Detection: faster aruco detection.
        :var int maxErroneousBitsInBorderRate: bits of the marker border should be black.
        :var int errorCorrectionRate: error correction rate (modified number of bits that can be corrected).
        :var int cornerRefinementMethod: method used to detect corners.
        :var int cornerRefinementWinSize: size of the window used to detect corners.
        :var int relativeCornerRefinmentWinSize: size of the window used to detect corners.
        :var int cornerRefinementMinAccuracy: minimum accuracy of the corner refinement process.
        :var int adaptiveThreshConstant: adaptive threshold value.

        :return: custom values for aruco parameters
    """

    aruco_params = cv2.aruco.DetectorParameters()

    if print_default_values:
        print('Detector parameters:')
        print(f'- minMarkerDistanceRate = {aruco_params.minMarkerDistanceRate}')
        print(f'- minMarkerPerimeterRate = {aruco_params.minMarkerPerimeterRate}')
        print(f'- minCornerDistanceRate = {aruco_params.minCornerDistanceRate}')
        print(f'- useAruco3Detection = {aruco_params.useAruco3Detection}')

        print('\nBitsExtraction parameters:')
        print(f'- minOtsuStdDev = {aruco_params.minOtsuStdDev}')

        print('\nMarker identification parameters:')
        print(f'- maxErroneousBitsInBorderRate = {aruco_params.maxErroneousBitsInBorderRate}')
        print(f'- errorCorrectionRate = {aruco_params.errorCorrectionRate}')

        print('\nCorner refinement parameters:')
        print(f'- cornerRefinementMethod = {aruco_params.cornerRefinementMethod}')
        print(f'- cornerRefinementWinSize = {aruco_params.cornerRefinementWinSize}')
        print(f'- relativeCornerRefinmentWinSize = {aruco_params.relativeCornerRefinmentWinSize}')
        print(f'- cornerRefinementMinAccuracy = {aruco_params.cornerRefinementMinAccuracy}')

        print(f'- adaptiveThreshConstant = {aruco_params.adaptiveThreshConstant}')

    '''
        parameter = custom value # default value
    '''
    aruco_params.minMarkerDistanceRate = 0.02  # 0.125
    aruco_params.minMarkerPerimeterRate = 0.02  # 0.03
    aruco_params.minCornerDistanceRate = 0.0002  # 0.05
    aruco_params.minOtsuStdDev = 0.1  # 5.0
    aruco_params.useAruco3Detection = False  # False
    aruco_params.maxErroneousBitsInBorderRate = 0.35  # 0.35
    aruco_params.errorCorrectionRate = 0.6  # 0.6
    aruco_params.cornerRefinementMethod = 0  # 0
    aruco_params.cornerRefinementWinSize = 2  # 5
    aruco_params.relativeCornerRefinmentWinSize = 0.3  # 0.3
    aruco_params.cornerRefinementMinAccuracy = 0.025  # 0.1
    aruco_params.adaptiveThreshConstant = 3  # 7.0

    if print_custom_values:
        print('Detector parameters:')
        print(f'- minMarkerDistanceRate = {aruco_params.minMarkerDistanceRate}')
        print(f'- minMarkerPerimeterRate = {aruco_params.minMarkerPerimeterRate}')
        print(f'- minCornerDistanceRate = {aruco_params.minCornerDistanceRate}')
        print(f'- useAruco3Detection = {aruco_params.useAruco3Detection}')

        print('\nBitsExtraction parameters:')
        print(f'- minOtsuStdDev = {aruco_params.minOtsuStdDev}')

        print('\nMarker identification parameters:')
        print(f'- maxErroneousBitsInBorderRate = {aruco_params.maxErroneousBitsInBorderRate}')
        print(f'- errorCorrectionRate = {aruco_params.errorCorrectionRate}')

        print('\nCorner refinement parameters:')
        print(f'- cornerRefinementMethod = {aruco_params.cornerRefinementMethod}')
        print(f'- cornerRefinementWinSize = {aruco_params.cornerRefinementWinSize}')
        print(f'- relativeCornerRefinmentWinSize = {aruco_params.relativeCornerRefinmentWinSize}')
        print(f'- cornerRefinementMinAccuracy = {aruco_params.cornerRefinementMinAccuracy}')

        print(f'- adaptiveThreshConstant = {aruco_params.adaptiveThreshConstant}')

    return aruco_params


if __name__ == '__main__':
    custom_aruco_params()
