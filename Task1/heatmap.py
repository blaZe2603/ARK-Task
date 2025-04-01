import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load images in grayscale
image_r = cv.imread("Task1/right.png", cv.IMREAD_GRAYSCALE)
image_l = cv.imread("Task1/left.png", cv.IMREAD_GRAYSCALE)

if image_r is None or image_l is None:
    raise ValueError("Error: One or both images could not be loaded. Check file paths.")

height, width = image_r.shape
heatmap = np.zeros((height - 10, width - 10), dtype="float32")

# Loop over pixels, avoiding edges
for i in range(5, height - 5):
    for j in range(5, width - 5):
        min_value = float('inf')
        best_k = j  # Start at current position

        # Searching in disparity range [-15, +15] pixels
        for k in range(max(5, j - 15), min(width - 5, j + 15)):
            # Extract 5x5 patches for comparison
            patch_l = image_l[i-2:i+3, j-2:j+3]
            patch_r = image_r[i-2:i+3, k-2:k+3]

            if patch_l.shape == patch_r.shape:
                diff = np.sum((patch_l.astype(np.float32) - patch_r.astype(np.float32)) ** 2)+10*abs(k - j)  # SSD metric
                if diff < min_value:
                    min_value = diff
                    best_k = k

        heatmap[i - 5, j - 5] = abs(best_k - j)  # Store disparity value

    print(f"Row {i}: {heatmap[i - 5, 10]:.2f}")  # Debug output

# Normalize disparity map for visualization
heatmap_norm = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX)
heatmap_uint8 = np.uint8(heatmap_norm)
colored_heatmap = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

# Show the heatmap
cv.imshow("Disparity Heatmap", colored_heatmap)
cv.waitKey(10000)
cv.destroyAllWindows()
