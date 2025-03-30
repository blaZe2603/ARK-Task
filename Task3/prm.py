import cv2 as cv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random
image = cv.imread("Task3/maze.png")
height, width , _  = image.shape
image = cv.imread("Task3/maze.png", cv.IMREAD_GRAYSCALE)
print(image.shape)
start_1 = (40,height-30)
end_1 = (100,height-30)
end_2 = (450,height-30)
NODES = 50
RADIUS = 100
RADIUS_SQ = RADIUS ** 2
node = np.random.randint([20,30],[width-20, height-30], size=(NODES, 2), dtype=np.int16)

node_possible = [start_1]



def squ_dist(p1, p2):
    return np.sum((p1 - p2) ** 2) 

def check_white_pixels(image, point1, point2):
    line_pixels = image.copy()
    cv.line(line_pixels, point1, point2, 128, 1) 
    line_points = cv.line(np.zeros_like(image), point1, point2, 128, 1)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if line_points[y,x] == 128:  
                if not image[y,x] == 255 and not image[y,x] == 128:
                    return False
    return True

def dist_check(image,p1, p2):
    if(check_white_pixels(image,p1,p2)):
        return squ_dist(p1, p2) < RADIUS_SQ
    else:
        return False  




for i in node:
    if (image[i[1], i[0]] == 255).all():
        cv.circle(image, tuple(i), 2, (128,0,0), -1)  
        node_possible.append(i)  


node_possible.append(end_1)  
node_possible = np.array(node_possible)
cv.circle(image,start_1,2,128,-1)
cv.circle(image,end_1,2,128,-1)

cv.imshow("image", image)
cv.waitKey(2000)
cv.destroyAllWindows()



for i in range(len(node_possible)):
    for j in range(i+1, len(node_possible)):  
        if dist_check(image,node_possible[i], node_possible[j]):
                cv.line(image, tuple(node_possible[i]), tuple(node_possible[j]), 128 , thickness=1, lineType=8, shift=0)
                
    # if(i%3 == 0):
    #     cv.imshow("image", image)
    #     cv.waitKey(1000)
    #     cv.destroyAllWindows()

cv.imshow("image", image)
cv.waitKey(5000)
cv.destroyAllWindows()