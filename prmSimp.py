import numpy as np
import cv2
from skimage.draw import line

img = cv2.imread(r"maze.jpg")
icolor = img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 1)

bg = 0 if np.mean(img)<120 else 1
n=5
si = 10
ei = 14
                 
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x][y] > 125:
            img[x][y] = 0 if bg else 255
        else:
            img[x][y] = 255 if bg else 0

points = []
nearest = []

t = int((img.shape[0])/n)
for x in range(n+1):
    cv2.line(icolor, (0,x*t+2), (img.shape[1]-1, x*t+2), (0,255,0), 1)
    cv2.line(icolor, (x*t+2,0), (x*t+2,img.shape[0]-1), (0,255,0), 1)

c = int(t/2)+2
for x in range(n):
    for y in range(n):
        points.append([c + y*t,c + x*t, len(points)])
        cv2.ellipse(icolor, (c + x*t,c + y*t), (2,2), 0,0,360,(255, 0, 255), -1)

for x in range(len(points)):
    nearest.append([])

for i in range(len(points)):
    for j in range(len(points)):
        if i == j:
            continue
        rr, cc = line(points[j][1], points[j][0], points[i][1], points[i][0])
        pixel_values = img[rr, cc]
        if np.any(pixel_values > 0):
            continue
        elif points[j][0] != points[i][0] and points[j][1] != points[i][1]:
            continue
        elif (points[j][1] == points[i][1] and abs(points[j][0] - points[i][0]) <120) or (points[j][0] == points[i][0] and abs(points[j][1] - points[i][1])<120):
            nearest[i].append([points[j][0],points[j][1], points[j][2]])
            cv2.line(icolor, (points[i][0], points[i][1]), (points[j][0], points[j][1]), (0,255,0), 1)

start = [points[si],points[si]]
end = points[ei]

out=[]
path = []
path.append(start)

counter = 0
while len(path):
    counter+=1
    current = path[0][0]
    if current==end:
        t = path.pop(0)
        out.append(t)
        break
    available = nearest[current[2]]
    rm=[]
    for j in range(len(available)):
        fl = 1
        for x in path:
            if x[0]==available[j]:
                fl=0
        for x in out:
            if x[0]==available[j]:
                fl=0
        if fl == 0:
            rm = [j] + rm
    for j in rm:
        available.pop(j)
    for i in range(len(available)):
        temp = []
        temp.append(available[i])
        temp.append(current)
        path.append(temp)
    t = path.pop(0)
    out.append(t)

cpoint = end
optimal = []
optimal = [cpoint] + optimal

while cpoint != start[0]:
    for x in out:
        if x[0] == cpoint:
            p = x[1]
            cv2.line(icolor, (cpoint[0], cpoint[1]), (p[0], p[1]), (255, 0, 0), 3)
            cv2.ellipse(icolor, (p[0],p[1]), (5,5), 0,0,360,(255,255,0), -1)
            cpoint = p
            optimal = [cpoint] + optimal

result = []
for x in optimal:
    result.append([x[0],x[1]])

print(result)

cv2.imshow("Maze", icolor)
cv2.waitKey(0)
cv2.destroyAllWindows()