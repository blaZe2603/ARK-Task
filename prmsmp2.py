import numpy as np
import cv2
from skimage.draw import line

img = cv2.imread(r"maze.jpg")
# img = cv2.resize(img, (int(img.shape[0]*2.5), int(img.shape[1]*2.5)))
icolor = img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 1)
                 
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x][y] > 125:
            img[x][y] = 255
        else:
            img[x][y] = 0

points = []
nearest = []

t = int((img.shape[0])/5)
for x in range(6):
    cv2.line(icolor, (0,x*t+2), (img.shape[1]-1, x*t+2), (0,255,0), 1)
    cv2.line(icolor, (x*t+2,0), (x*t+2,img.shape[0]-1), (0,255,0), 1)

c = int(t/2)+2
for x in range(5):
    for y in range(5):
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

# print(points)

# for i in range(len(points)):
#     for j in range(len(points)):
#         if i == j:
#             continue
#         rr, cc = line(points[j][1], points[j][0], points[i][1], points[i][0])
#         pixel_values = img[rr, cc]
#         print(pixel_values[0])
#         if np.any(pixel_values[0] >0):
#             continue
#         else:
#             nearest[i].append([points[j][0],points[j][1], points[j][2]])
#     print(len(nearest[i]))
# print(nearest)

# for i in range(len(points)):
#     u = points[i-10] if i>10 else 0
#     d = points[i+10] if i<len(points)-10 else 0
#     l = points[i-1] if i>1 else 0
#     r = points[i+1] if i<len(points)-1 else 0
#     print(u,d,l,r)
    # fu=1
    # if u:
    #     print(u)
    #     print(points[i])
    #     for x in range(u[1], points[i][1]):
    #         print(img[u[0],x])
    #         if img[u[0],x] == 255:
    #             fu=0
    # else:
    #     fu = 0
    # if fu: 
    #     nearest[i].append([u[0],u[1], u[2]])

for i in range(len(points)):
    p1 = points[i]
    for y in nearest[i]:
        cv2.line(icolor, (p1[0], p1[1]), (y[0],y[1]), (0,0,255), 1)

# print(poaints)

start = [[54, 262, 10],[54, 262, 10]]
# start = [54, 262, 10]
end = [470, 262, 14]

cv2.ellipse(icolor, (start[0][0], start[0][1]), (3,3), 0,0,360,(255,0,0),-1)
out=[]
path = []
path.append(start)

counter = 0
while len(path):
    print("Current:")
    print(counter)
    counter+=1
    current = path[0][0]
    print(path)

    # print(current)
    available = nearest[current[2]]
    
    rm=[]
    # print(available)
    for j in range(len(available)):
        # print(j)
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
    print("Available:")
    print(available)
    
    # print("Current:")
    # print(current)
    for i in range(len(available)):
        temp = []
        temp.append(available[i])
        temp.append(current)
        # print(temp)
        path.append(temp)
    t = path.pop(0)
    out.append(t)
    # print(path)


# print(img.shape)
cv2.imshow("Maze", icolor)
cv2.imshow("BW", img)
cv2.waitKey(0)
cv2.destroyAllWindows()