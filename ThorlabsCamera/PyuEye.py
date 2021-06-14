from pyueye import ueye
import numpy as np
import cv2
import sys



h_cam = ueye.HIDS(0)
ret = ueye.is_InitCamera(h_cam, None)

if ret != ueye.IS_SUCCESS:
    pass
