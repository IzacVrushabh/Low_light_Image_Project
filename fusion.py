from image_enhancement import image_enhancement
import cv2 as cv
from contrast_image import contrast_image
from contrast_image import quantitation
import skimage.measure
import numpy as np
import glob
import csv
import os
import imquality.brisque as brisque
import PIL.Image
import warnings
from skimage import metrics

warnings.filterwarnings('ignore')

p = 0.7
q = 0.3

path_list = []
names_list = []

folder='D:\Android_Projects\ExDarkImages\Mix-Images'

for f in glob.glob(folder+'/*.jpg'):
    names_list.append(os.path.split(f)[-1])
# print(names_list)

for f in glob.glob(folder+'/*.jpg'):
    path_list.append(f)

# print(path_list)
read_images = []
for image in path_list:
    read_images.append(cv.imread(image))
# image = cv.imread("image1.jpg")

Avg_Entropy = []
for image in read_images:
    ie = image_enhancement.IE(image, color_space = 'RGB')
    result1 = ie.BBHE() #a
    result2 = ie.BPHEME() #b

    r1, g1, b1 = cv.split(result1)
    # cv.imshow("Input-Image", image)
    BBHE_image = (np.dstack((r1*q, g1*q, b1*q))).astype(np.uint8)
    # cv.imshow("BBHE-Output", BBHE_image)
    r2, g2, b2 = cv.split(result2)
    BPHEME_image = (np.dstack((r2*p, g2*p, b2*p))).astype(np.uint8)
    # print(rgb_uint8_2)
    # cv.imshow("BPHEME-Output", BPHEME_image)

    ent2 = skimage.measure.shannon_entropy(image)
    print("Original Entropy", ent2)
    # cv.imshow("Output", (p*result2) + (q*result1))
    ent1 = skimage.measure.shannon_entropy(BBHE_image + BPHEME_image)
    Avg_Entropy.append(ent1)
    # print(ent2)
    # cv.imshow("Fusion-Output", BBHE_image + BPHEME_image)
    print("Final Entropy", ent1)

print("{:.3f}".format(np.array(Avg_Entropy).mean()))


cv.waitKey(0)
cv.destroyAllWindows()


# op = p*a + q*b
# p*a + (1-p)*b
# b + (a - b)*p  # p = 0.3
# 6.077 + (0.139)*0.3 = 6.1187