@author: valen
"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
''' Phase Correlation based image matching and registration libraries
'''
__author__ = "Yoshi Ri"
__copyright__ = "Copyright 2017, The University of Tokyo"
__credits__ = ["Yoshi Ri"]
__license__ = "BSD"
__version__ = "1.0.1"
__maintainer__ = "Yoshi Ri"
__email__ = "yoshiyoshidetteiu@gmail.com"
__status__ = "Production"

# Phase Correlation to Estimate Pose
import cv2
import numpy as np
import matplotlib.pyplot as plt  # matplotlibの描画系
import math
import sys


class imregpoc:
    def __init__(self, iref, icmp, *, threshold=0.06, alpha=0.5, beta=0.8, fitting='WeightedCOG'):
        self.orig_ref = iref.astype(np.float32)
        self.orig_cmp = icmp.astype(np.float32)
        self.th = threshold
        self.orig_center = np.array(self.orig_ref.shape) / 2.0
        self.alpha = alpha
        self.beta = beta
        self.fitting = fitting

        self.param = [0, 0, 0, 1]
        self.peak = 0
        self.affine = np.float32([1, 0, 0, 0, 1, 0]).reshape(2, 3)
        self.perspective = np.float32([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3, 3)

        # set ref, cmp, center
        self.fft_padding()
        self.match()

    def define_fftsize(self):
        refshape = self.orig_ref.shape
        cmpshape = self.orig_cmp.shape
        if not refshape == cmpshape:
            print("The size of two input images are not equal! Estimation could be inaccurate.")
        maxsize = max(max(refshape), max(cmpshape))
        # we can use faster fft window size with scipy.fftpack.next_fast_len
        return maxsize

    def padding_image(self, img, imsize):
        pad_img = np.pad(img, [(0, imsize[0] - img.shape[0]), (0, imsize[1] - img.shape[1])], 'constant')
        return pad_img

    def fft_padding(self):
        maxsize = self.define_fftsize()
        self.ref = self.padding_image(self.orig_ref, [maxsize, maxsize])
        self.cmp = self.padding_image(self.orig_cmp, [maxsize, maxsize])
        self.center = np.array(self.ref.shape) / 2.0

    def fix_params(self):
        # If you padded to right and lower, perspective is the same with original image
        self.param = self.warp2poc(perspective=self.perspective, center=self.orig_center)

    def match(self):
        height, width = self.ref.shape
        self.hanw = cv2.createHanningWindow((width, height), cv2.CV_64F)

        # Windowing and FFT
        G_a = np.fft.fft2(self.ref * self.hanw)
        G_b = np.fft.fft2(self.cmp * self.hanw)

        # 1.1: Frequency Whitening
        self.LA = np.fft.fftshift(np.log(np.absolute(G_a) + 1))
        self.LB = np.fft.fftshift(np.log(np.absolute(G_b) + 1))
        # 1.2: Log polar Transformation
        cx = self.center[1]
        cy = self.center[0]
        self.Mag = width / math.log(width)
        self.LPA = cv2.logPolar(self.LA, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        self.LPB = cv2.logPolar(self.LB, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        # 1.3:filtering
        LPmin = math.floor(self.Mag * math.log(self.alpha * width / 2.0 / math.pi))
        LPmax = min(width, math.floor(self.Mag * math.log(width * self.beta / 2)))
        assert LPmax > LPmin, 'Invalid condition!\n Enlarge lpmax tuning parameter or lpmin_tuning parameter'
        Tile = np.repeat([0.0, 1.0, 0.0], [LPmin - 1, LPmax - LPmin + 1, width - LPmax])
        self.Mask = np.tile(Tile, [height, 1])
        self.LPA_filt = self.LPA * self.Mask
        self.LPB_filt = self.LPB * self.Mask

        # 1.4: Phase Correlate to Get Rotation and Scaling
        Diff, peak, self.r_rotatescale = self.PhaseCorrelation(self.LPA_filt, self.LPB_filt)
        theta1 = 2 * math.pi * Diff[1] / height;  # deg
        theta2 = theta1 + math.pi;  # deg theta ambiguity
        invscale = math.exp(Diff[0] / self.Mag)
        # 2.1: Correct rotation and scaling
        b1 = self.Warp_4dof(self.cmp, [0, 0, theta1, invscale])
        b2 = self.Warp_4dof(self.cmp, [0, 0, theta2, invscale])

        # 2.2 : Translation estimation
        diff1, peak1, self.r1 = self.PhaseCorrelation(self.ref, b1)  # diff1, peak1 = PhaseCorrelation(a,b1)
        diff2, peak2, self.r2 = self.PhaseCorrelation(self.ref, b2)  # diff2, peak2 = PhaseCorrelation(a,b2)
        # Use cv2.phaseCorrelate(a,b1) because it is much faster

        # 2.3: Compare peaks and choose true rotational error
        if peak1 > peak2:
            Trans = diff1
            peak = peak1
            theta = -theta1
        else:
            Trans = diff2
            peak = peak2
            theta = -theta2

        if theta > math.pi:
            theta -= math.pi * 2
        elif theta < -math.pi:
            theta += math.pi * 2

        # Results
        self.param = [Trans[0], Trans[1], theta, 1 / invscale]
        self.peak = peak
        self.perspective = self.poc2warp(self.center, self.param)
        self.affine = self.perspective[0:2, :]
        self.fix_params()

    def match_new(self, newImg):
        self.cmp_orig = newImg
        self.fft_padding()
        height, width = self.cmp.shape
        cy, cx = height / 2, width / 2
        G_b = np.fft.fft2(self.cmp * self.hanw)
        self.LB = np.fft.fftshift(np.log(np.absolute(G_b) + 1))
        self.LPB = cv2.logPolar(self.LB, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        self.LPB_filt = self.LPB * self.Mask
        # 1.4: Phase Correlate to Get Rotation and Scaling
        Diff, peak, self.r_rotatescale = self.PhaseCorrelation(self.LPA_filt, self.LPB_filt)
        theta1 = 2 * math.pi * Diff[1] / height;  # deg
        theta2 = theta1 + math.pi;  # deg theta ambiguity
        invscale = math.exp(Diff[0] / self.Mag)
        # 2.1: Correct rotation and scaling
        b1 = self.Warp_4dof(self.cmp, [0, 0, theta1, invscale])
        b2 = self.Warp_4dof(self.cmp, [0, 0, theta2, invscale])

        # 2.2 : Translation estimation
        diff1, peak1, self.r1 = self.PhaseCorrelation(self.ref, b1)  # diff1, peak1 = PhaseCorrelation(a,b1)
        diff2, peak2, self.r2 = self.PhaseCorrelation(self.ref, b2)  # diff2, peak2 = PhaseCorrelation(a,b2)
        # Use cv2.phaseCorrelate(a,b1) because it is much faster

        # 2.3: Compare peaks and choose true rotational error
        if peak1 > peak2:
            Trans = diff1
            peak = peak1
            theta = -theta1
        else:
            Trans = diff2
            peak = peak2
            theta = -theta2

        if theta > math.pi:
            theta -= math.pi * 2
        elif theta < -math.pi:
            theta += math.pi * 2

        # Results
        self.param = [Trans[0], Trans[1], theta, 1 / invscale]
        self.peak = peak
        self.perspective = self.poc2warp(self.center, self.param)
        self.affine = self.perspective[0:2, :]
        self.fix_params()

    def poc2warp(self, center, param):
        cx, cy = center
        dx, dy, theta, scale = param
        cs = scale * math.cos(theta)
        sn = scale * math.sin(theta)

        Rot = np.float32([[cs, sn, 0], [-sn, cs, 0], [0, 0, 1]])
        center_Trans = np.float32([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        center_iTrans = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        cRot = np.dot(np.dot(center_Trans, Rot), center_iTrans)
        Trans = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        Affine = np.dot(cRot, Trans)
        return Affine

    def warp2poc(self, center, perspective):
        cx, cy = center
        Center = np.float32([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        iCenter = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

        pocmatrix = np.dot(np.dot(iCenter, perspective), Center)
        dxy = np.dot(np.linalg.inv(pocmatrix[0:2, 0:2]), pocmatrix[0:2, 2])
        scale = np.sqrt(pocmatrix[0, 0] ** 2 + pocmatrix[0, 1] ** 2)
        theta = np.arctan2(pocmatrix[0, 1], pocmatrix[0, 0])
        return [dxy[0], dxy[1], theta, scale]

    # Waro Image based on poc parameter
    def Warp_4dof(self, Img, param):
        center = np.array(Img.shape) / 2
        rows, cols = Img.shape
        Affine = self.poc2warp(center, param)
        outImg = cv2.warpPerspective(Img, Affine, (cols, rows), cv2.INTER_LINEAR)
        return outImg

    def SubpixFitting(self, mat):
        if self.fitting == 'COG':
            TY, TX = self.CenterOfGravity(mat)
        elif self.fitting == 'WeightedCOG':
            TY, TX = self.WeightedCOG(mat)
        else:
            print("Undefined subpixel fitting method! Use weighted center of gravity method instead.")
            TY, TX = self.WeightedCOG(mat)

        return [TY, TX]

    # Get peak point
    def CenterOfGravity(self, mat):
        hei, wid = mat.shape
        if hei != wid:  # if mat size is not square, there must be something wrong
            print("Skip subpixel estimation!")
            return [0, 0]
        Tile = np.arange(wid, dtype=float) - (wid - 1.0) / 2.0
        Tx = np.tile(Tile, [hei, 1])  # Ty = Tx.T
        Sum = np.sum(mat)
        # print(mat)
        Ax = np.sum(mat * Tx) / Sum
        Ay = np.sum(mat * Tx.T) / Sum
        return [Ay, Ax]

    # Weighted Center Of Gravity
    def WeightedCOG(self, mat):
        if mat.size == 0:
            print("Skip subpixel estimation!")
            Res = [0, 0]
        else:
            peak = mat.max()
            newmat = mat * (mat > peak / 10)  # discard information of lower peak
            Res = self.CenterOfGravity(newmat)
        return Res

    # Phase Correlation
    def PhaseCorrelation(self, a, b):
        height, width = a.shape
        # dt = a.dtype # data type
        # Windowing

        # FFT
        G_a = np.fft.fft2(a * self.hanw)
        G_b = np.fft.fft2(b * self.hanw)
        conj_b = np.ma.conjugate(G_b)
        R = G_a * conj_b
        R /= np.absolute(R)
        r = np.fft.fftshift(np.fft.ifft2(R).real)
        # Get result and Interpolation
        DY, DX = np.unravel_index(r.argmax(), r.shape)
        # Subpixel Accuracy
        boxsize = 5
        box = r[DY - int((boxsize - 1) / 2):DY + int((boxsize - 1) / 2) + 1,
              DX - int((boxsize - 1) / 2):DX + int((boxsize - 1) / 2) + 1]  # x times x box
        # subpix fitting
        self.peak = r[DY, DX]
        TY, TX = self.SubpixFitting(box)
        sDY = TY + DY
        sDX = TX + DX
        # Show the result
        return [math.floor(width / 2) - sDX, math.floor(height / 2) - sDY], self.peak, r


    def getParam(self):
        return self.param

    def getPeak(self):
        return self.peak

    def getAffine(self):
        return self.affine

    def getPerspective(self):
        return self.perspective

    def showRotatePeak(self):
        plt.imshow(self.r_rotatescale, vmin=self.r_rotatescale.min(), vmax=self.r_rotatescale.max(), cmap='gray')
        plt.show()

    def showTranslationPeak(self):
        plt.subplot(211)
        plt.imshow(self.r1, vmin=self.r1.min(), vmax=self.r1.max(), cmap='gray')
        plt.subplot(212)
        plt.imshow(self.r2, vmin=self.r2.min(), vmax=self.r2.max(), cmap='gray')
        plt.show()

#looping through directory 
import csv
import os
import glob
from collections import OrderedDict
from datetime import datetime
from csv import writer
from csv import reader
from csv import writer
from csv import reader
from scipy import ndimage


directory = r"C:\Users\valmzztt.stu\My Python Scripts\Thorlabs Camera\Images\Images_2106\PatternDetection_y_Trial1"

# creating the dictionary to register the real displacement versus pixel offset
dicts = {}
keys = []
values = []
index = 0

for filename in os.listdir(directory):
    if filename.startswith("image1_"):
        input_path = os.path.join(directory, filename)
        ref = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        ref_laplacian3 = ndimage.gaussian_laplace(ref, sigma=3)

#looping thorugh directory while calculating the laplacian image for edge detection and then comparing it with image1 in the folder for subpixel shift

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        input_path = os.path.join(directory, filename)
        cmp = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        cmp_laplacian3= ndimage.gaussian_laplace(cmp, sigma=3)
        # reference parameter set to 'WeightedCOG'
        match = imregpoc(ref_laplacian3, cmp_laplacian3)
        x_offset = match.param[0]
        values.append(x_offset)
        print(x_offset)
        index = index +1


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
    
with open('pixel_vs_displ.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\r')
    #writer = csv.writer(output, delimiter=' ', lineterminator='\r')
    for key, value in dict1.items():
        writer.writerow([key, value])
