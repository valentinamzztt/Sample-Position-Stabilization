# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:12:45 2021
Taken from Python-Computer-Vision-Tutorials-Image-Fourier-Transform-part-4.1-Motion-Detection-/motion_detection_with_fourier_transform01.py
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
for i in range(10): plt.close()




image1_filename = r"C:\Users\valen\My Python Scripts\Quantum Internship\IMAGE ARPES\fc2_save_2021-04-17-101329-0000.png"
image2_filename = r"C:\Users\valen\My Python Scripts\Quantum Internship\IMAGE ARPES\fc2_0000_shifted.png"
image1 = cv2.imread(image1_filename)[:,:,0]
image2 = cv2.imread(image2_filename)[:,:,0]

img = cv2.imread(r"C:\Users\valen\My Python Scripts\Quantum Internship\IMAGE ARPES\fc2_save_2021-04-17-101329-0000.png",0)
rows,cols = img.shape

nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)

crop_size = nrows - 1
image1_cp = image1[:crop_size, :crop_size]
image2_cp = image2[:crop_size, :crop_size]



f1 = cv2.dft(image1_cp.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
f2 = cv2.dft(image2_cp.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

f1_shf = np.fft.fftshift(f1)
f2_shf = np.fft.fftshift(f2)

f1_shf_cplx = f1_shf[:,:,0]*1j + 1*f1_shf[:,:,1]
f2_shf_cplx = f2_shf[:,:,0]*1j + 1*f2_shf[:,:,1]

f1_shf_abs = np.abs(f1_shf_cplx)
f2_shf_abs = np.abs(f2_shf_cplx)
total_abs = f1_shf_abs * f2_shf_abs

P_real = (np.real(f1_shf_cplx)*np.real(f2_shf_cplx) +
          np.imag(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
P_imag = (np.imag(f1_shf_cplx)*np.real(f2_shf_cplx) +
          np.real(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
P_complex = P_real + 1j*P_imag

P_inverse = np.abs(np.fft.ifft2(P_complex))


max_id = [0, 0]
max_val = 0
for idy in range(crop_size):
    for idx in range(crop_size):
        if P_inverse[idy,idx] > max_val:
            max_val = P_inverse[idy,idx]
            max_id = [idy, idx]
shift_x = max_id[0]
shift_y = max_id[1]
shifted = (shift_x, shift_y)
print (f"Detected subpixel offset (x, y): {shifted}")
