import cv2
import math
import time
import numpy as np
import os
import csv
import glob
from scipy import ndimage

class CorrelationCalculator(object):
    'TODO: class description'

    version = '0.1'

    def __init__(self, initial_frame, detection_threshold=4):
        self.initial_frame = np.float32(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY))
        self.detection_threshold = detection_threshold

    def detect_phase_shift(self, current_frame):
        'returns detected sub-pixel phase shift between two arrays'
        self.current_frame = np.float32(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
        shift = cv2.phaseCorrelate(self.initial_frame, self.current_frame)
        return shift


# implementation
directory = r"C:\Users\valmzztt.stu\My Python Scripts\Thorlabs Camera\Images\Images_2106\PatternDetection_y_Trial1"

dict1 = {}
values = []

for filename in os.listdir(directory):
    if filename.startswith("image1_"):
        input_path = os.path.join(directory, filename)
        img = cv2.imread(input_path)
        img_laplacian = ndimage.gaussian_laplace(img, sigma=3)

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        index +=1
        input_path = os.path.join(directory, filename)
        img2 = cv2.imread(input_path)
        img2_laplacian = ndimage.gaussian_laplace(img2, sigma=3)
        obj = CorrelationCalculator(img_laplacian)
        shift = obj.detect_phase_shift(img2_laplacian)
        x_transl = shift[0][0]
        print(x_transl)
        error = shift[1]
        error_values.append(error)
        values.append(x_transl)

keys = range(1, index)
keys = [int(i) for i in keys]
dict1 = dict(zip(keys, values))


with open('pixel_vs_displ.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\r')
    for key, value in dict1.items():
        writer.writerow([key, value])

dicts = {}
keys = []

filenames = [os.path.basename(x) for x in glob.glob((os.path.join(directory, '*.png')))]
for filename in filenames:
    new_filename = filename[5:8]
    if new_filename[2] == "_":
        new_filename = new_filename[0:2]
        keys.append(new_filename)
    elif new_filename[1] == "_":
        new_filename = new_filename[0:1]
        keys.append(new_filename)
    else:
        keys.append(new_filename)

keys = [int(i) for i in keys]
dict1 = dict(zip(keys, values))
dict1 = dict(sorted(dict1.items()))
new_values = []

for key, value in dict1.items():
    new_values.append(value)
for key, value in dict3.items():
    new_error_values.append(value)

with open('pixel_vs_displ.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\r')
    #writer = csv.writer(output, delimiter=' ', lineterminator='\r')
    for key, value in dict1.items():
        writer.writerow([key, value])


