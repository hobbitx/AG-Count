import numpy as np
import pandas as pd
import cv2 as cv
import sys
import os
import imutils
from matplotlib import pyplot as plt

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    #cv.imshow("Display window", sharpened)
    return sharpened

def getCount(image,treshold,contrast):
    
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    new_image = cv.addWeighted( image, alpha_c, image, 0, gamma_c) #add contrast for image
    image = unsharp_mask(new_image) 
    image_blur_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_res ,image_thresh = cv.threshold(image_blur_gray,treshold,255,cv.THRESH_BINARY_INV) #get threshold image

    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(image_thresh,cv.MORPH_OPEN,kernel) 

    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5) 
    _, last_image =  cv.threshold(dist_transform, 0.3*dist_transform.max(),255,0)

    last_image2 = np.uint8(last_image)
    cnts = cv.findContours(last_image2.copy(), cv.RETR_EXTERNAL,	cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return len(cnts)


def check_neuron(path,treshold, contrast):
    for str_file in os.listdir(path):
        filename = f"{path}/{str_file}"
        image = cv.imread(filename)
        count = getCount(image,treshold, contrast)
        print(filename)
        print(count)

dt = pd.read_json("inputs.json")
print(dt)
paths = dt.keys()
for path in paths:
    end_path = "Imagens/"+path
    check_neuron(end_path,210,63)