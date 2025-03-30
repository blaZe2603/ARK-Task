import cv2
import numpy as np
import random
import math
def color_near(color1, color2):
    x1=int(color1[0])
    y1=int(color1[1])
    z1=int(color1[2])
    x2=int(color2[0])
    y2=int(color2[1])
    z2=int(color2[2])
    if (x1-x2)*2+(y1-y2)*2+(z1-z2)*2>3000:
        return 0
    return 1
def pre_process_image(img):
    height, width=img.shape[:2]
    color2=[159,82,67]
    for i in range(height):
        for j in range(width):
            if color_near(img[i][j],color2):
                img[i][j]=[255,255,255]
            else:
                img[i][j]=[0,0,0]
    return img
img=cv2.imread("C:\\Users\\Keshav\\OneDrive\\Documents\\ARK\\Task3\\maze.png")
img1=img.copy()
processed_img=pre_process_image(img)
start=[156,637]
end=[351,505]
# img1[start[0]][start[1]]=[0,0,0]
cv2.imshow("namE", img1)
cv2.waitKey(0)
number=0
nodes_lst=[]            #format for each node is [node,parent]
nodes_lst.append([start,start])
height,width=img.shape[:2]

def cartesian_dist(point1,point2):
    return math.sqrt((point1[0]-point2[0])*2+(point1[1]-point2[1])*2)

def closest(node, nodes_lst):
    min_=float('inf')
    closest_point=None
    for point in nodes_lst:
        temp=cartesian_dist(node,point[0])
        if temp<min_:
            min_=temp
            closest_point=point[0]
    return closest_point

max_dist=5

def reduce_dist_to_max(node,point,max_dist):
    x1,y1=node
    x2,y2=point
    if x1!=x2:
        tan=(y2-y1)/(x2-x1)
        sin=math.sqrt(abs(tan)/(math.sqrt(1+tan**2)))
        if tan>0:
            cos=math.sqrt(1-sin**2)
        else:
            cos=-math.sqrt(1-sin**2)
    else:
        sin,cos=1,0
    if y2<y1:
        r=max_dist
    else:
        r=-max_dist
    return [int(x2+r*cos), int(y2+r*sin)]

def bresenham_line(x0, y0, x1, y1):     #from GPT
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    del_x = 1 if x0 < x1 else -1
    del_y = 1 if y0 < y1 else -1
    err = dx - dy

    while(1):
        points.append([x0, y0])
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += del_x
        if e2 < dx:
            err += dx
            y0 += del_y

    return points

while(number<700):
    node=[random.randint(0,height-1), random.randint(0,width-1)]
    if processed_img[node[0]][node[1]][0]==0:
        continue
    closest_point=closest(node,nodes_lst)
    if cartesian_dist(node,closest_point)>max_dist:
        node=reduce_dist_to_max(node,closest_point,max_dist)
    line_points=bresenham_line(node[0], node[1], closest_point[0], closest_point[1])
    flag=0
    for points in line_points:
        if processed_img[points[0]][points[1]][0]==0:
            flag=1
    if flag:
        continue
    number+=1
    nodes_lst.append([node,closest_point])
    cv2.line(img1, [node[1],node[0]], [closest_point[1],closest_point[0]], (0, 255, 0), 3)
    cv2.imshow("Line Drawing", img1)
    cv2.waitKey(20)

closest_to_end=closest(end,nodes_lst)
if cartesian_dist(closest_to_end,end)>10:
    print("not found")
    exit(10)
nodes_lst.append([end,closest_to_end])

def find_min_len_path(goal,visited):            #min_dist and path is returned
    if goal[0]==start[0] and goal[1]==start[1]:
        return 0,[[start[0],start[1]]]
    min_dist=float('inf')
    min_dist_path=None
    ind=-1
    for node in nodes_lst:
        ind+=1
        if ind in visited:
            continue
        if node[0][0]==goal[0] and node[0][1]==goal[1]:
            visited.append(ind)
            until_parent=find_min_len_path(node[1],visited)
            path_len=cartesian_dist(goal,node[1])+until_parent[0]
            if path_len<min_dist:
                min_dist=path_len
                x=until_parent[1][:]
                x.append(goal)
                min_dist_path=x
    return min_dist, min_dist_path

final_route=find_min_len_path(end,[])[1]
for i in range(len(final_route)-1):
    cv2.line(img1, [final_route[i][1],final_route[i][0]], [final_route[i+1][1],final_route[i+1][0]], (255, 0, 0), 3)
cv2.imshow("final route", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
print((find_min_len_path(end,[]))[1])