import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load images in grayscale
image_r = cv.imread("Task1/right.png", cv.IMREAD_GRAYSCALE)
image_l = cv.imread("Task1/left.png", cv.IMREAD_GRAYSCALE)


SIZE = 5
DISPARITY_RANGE = 15
EPSILON = 50
height, width = image_r.shape
heatmap = np.zeros((height - 2*SIZE, width - 2*SIZE), dtype="float32")


for i in range(SIZE, height - SIZE):
    for j in range(SIZE, width - SIZE):
        min_value = float('inf')
        best_k = j   # Initialize best match to current pixel

        for k in range(max(SIZE, j - DISPARITY_RANGE), min(width - SIZE, j + DISPARITY_RANGE)):
            patch_l = image_l[i-2:i+3, j-2:j+3]
            patch_r = image_r[i-2:i+3, k-2:k+3]

            diff = np.sum((patch_l.astype(np.float32) - patch_r.astype(np.float32)) ** 2) 
            if diff < min_value:
                min_value = diff
                best_k = k

        heatmap[i - SIZE, j - SIZE] = 1/(abs(best_k - j) + EPSILON )

    print(f"Row {i}: {heatmap[i - 5, 10]:.2f}")  

# Normalize disparity map for visualization
#then showing heatmap
heatmap_norm = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX)
heatmap_uint8 = np.uint8(heatmap_norm)
colored_heatmap = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

# Show the heatmap
cv.imshow("Disparity Heatmap", colored_heatmap)
cv.waitKey(10000)
cv.destroyAllWindows()