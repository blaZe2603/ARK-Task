import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load images in grayscale
image_r = cv.imread("Task1/right.png", cv.IMREAD_GRAYSCALE)
image_l = cv.imread("Task1/left.png", cv.IMREAD_GRAYSCALE)
