import cv2 as cv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random

image_r = cv.imread("Task1/right.png")
image_l = cv.imread("Task1/left.png")
height, width , _  = image_r.shape
cout = 0
for i in range(height):
    for j in range(width):
        if (image_r[i, j] != image_l[i, j]).any():
            # print([i,j])
            cout = cout+ 1

print(image_l[(1,1)])
print(image_r[(1,1)])
print(height*width-cout)