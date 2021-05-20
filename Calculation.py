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



imagenames_list = []
csvName = "CUP"

folder='D:\Android_Projects\ExDarkImages\Cup'

names_list = []
for f in glob.glob(folder+'/*.jpg'):
    names_list.append(os.path.split(f)[-1])
#print(names_list)s

for f in glob.glob(folder+'/*.jpg'):
    imagenames_list.append(f)

#print(imagenames_list)



# Value Array
read_images = []
entropy_list = []


for image in imagenames_list:
    read_images.append(cv.imread(image))

output_images = []
for image in read_images:
    ie = image_enhancement.IE(image, color_space = 'HSV')
    output_images.append(ie.BPHEME())
    entropy_list.append(skimage.measure.shannon_entropy(image))

# PSNR list
psnr_list = []
for (ip,op) in zip(read_images,output_images):
    psnr_list.append(metrics.peak_signal_noise_ratio(ip,op))


# brisque list
brisque_list = []
for image in imagenames_list:
    img = PIL.Image.open(image)
    brisque_list.append(brisque.score(img))





with open('BPHEME_'+ csvName +'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["SN", "Name", "Entropy", "BRISQUE", "PSNR"])

    count = 0
    for (names, ent, brsq, psn) in zip(names_list, entropy_list, brisque_list, psnr_list):

        writer.writerow([count+1, names, ent, brsq, psn])
        count = count+1





#cv.imshow("input",input)
#cv.imshow("output",output)
cv.waitKey(0)
cv.destroyAllWindows()