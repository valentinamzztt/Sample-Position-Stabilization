# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:43:04 2021

@author: valen
"""
import matplotlib.pyplot as plt
import cv2
from skimage import color, io, data
from skimage.io import imread, imshow
from skimage.registration import phase_cross_correlation 
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift, shift


image = io.imread(r"C:\Users\valen\My Python Scripts\Quantum Internship\Shifted_image\shifted1.png")
offset_image = io.imread(r"C:\Users\valen\My Python Scripts\Quantum Internship\Shifted_image\shifted2.png")
dim = image.shape

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
offset_image =cv2.cvtColor(offset_image, cv2.COLOR_BGR2GRAY) 

#resize offset image to get same shape
offset_image = resize(offset_image, (image.shape[0], image.shape[1]),
                       anti_aliasing=True)

# subpixel precision
#Upsample factor 100 = images will be registered to within 1/100th of a pixel.
#Default is 1 which means no upsampling.  
shifted, error, diffphase = phase_cross_correlation(image, offset_image)
print(f"Detected subpixel offset (y, x): {shifted}")


#optional: plotting corrected image for translation
corrected_image = shift(offset_image, shift=(shifted[0], shifted[1]), mode='constant')

plt.imshow(corrected_image)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()











