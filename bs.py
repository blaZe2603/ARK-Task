import numpy as np
import cv2
import matplotlib.pyplot as plt


class ImageEnvironment:
    def __init__(self, image):
        # The image should be a binary image, where white is free space (255) and black is obstacle (0)
        self.image = image

    def check_collision(self, point):
        """
        Check if the pixel at the given point is an obstacle (not white).
        point: (x, y) -> The pixel coordinates to check
        """
        x, y = point
        if self.image[y, x] != 255:  # If the pixel is not white, it's an obstacle
            return True
        return False

    def check_collision_line(self, p1, p2, step_size=0.1):
        """
        Check if the line between points p1 and p2 collides with any obstacles in the image.
        p1, p2: (x, y) points
        step_size: Distance between each sampled point along the line segment
        """
        # Calculate the number of steps to take based on the distance between p1 and p2
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        steps = int(dist / step_size)

        # Sample points along the line and check for collisions
        for i in range(steps + 1):
            t = i / steps
            # Linearly interpolate between p1 and p2
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))

            # Check if this point collides with an obstacle
            if self.check_collision((x, y)):
                return True  # Collision detected

        return False  # No collision found

    def draw(self, path=[]):
        """
        Visualize the environment and the path on the image.
        """
        img_copy = self.image.copy()
        # Draw the path (if any) on the image
        if path:
            for i in range(len(path) - 1):
                cv2.line(img_copy, tuple(path[i]), tuple(path[i + 1]), (255, 0, 0), 2)  # Red path
        plt.imshow(img_copy, cmap='gray')
        plt.show()


# Function to check if a path is collision-free
def is_path_collision_free(env, path, step_size=0.1):
    """
    Check if a path (list of points) is collision-free on the image.
    env: The ImageEnvironment object with the image
    path: List of points [(x1, y1), (x2, y2), ...] representing the path
    step_size: Distance between each sampled point along each line segment
    """
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        if env.check_collision_line(p1, p2, step_size):
            return False  # If any part of the path collides, return False
    return True  # No collisions found


# Example: Create a simple binary image with obstacles (black)
image = np.ones((200, 200), dtype=np.uint8) * 255  # White background (free space)

# Add some obstacles (black squares)
cv2.rectangle(image, (50, 50), (100, 100), 0, -1)  # Obstacle 1
cv2.rectangle(image, (150, 100), (180, 130), 0, -1)  # Obstacle 2
cv2.circle(image, (100, 150), 20, 0, -1)  # Obstacle 3 (circle)

# Create environment with the image
env = ImageEnvironment(image)

# Example Path (a series of points)
path = [(10, 10), (50, 50), (150, 110), (190, 190)]

# Check if the path is collision-free
is_collision_free = is_path_collision_free(env, path, step_size=0.5)

# Output the result
print("Is the path collision-free?", is_collision_free)

# Visualize the environment and path
env.draw(path)
