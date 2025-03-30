def check_white_pixels(image, point1, point2):

#     line_pixels = image.copy()
#     cv.line(line_pixels, point1, point2, 255, 1)
#     for x in range(image.shape[1]):
#         for y in range(image.shape[0]):
#             if line_pixels[y, x].all() == 255:  # Only check the pixels that are part of the line
#                 print("yea")
#                 if image[y, x].all() != 255:  # If any of these pixels are not white, return False
#                     return False
#     return True