import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes,
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output


def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    '''
    :param image: Image to draw boxes upon (3 color channels)
    :param bboxes: Bounding boxes defining two corners ((x1, y1), (x2, y2))
    :param color: RGB color value of box edges
    :param thick: thickness in pixels of box borders
    :return: Drawn up copy of original image

    Takes image and list of bounding boxes and returns a copy of the
     image with the bounding boxes drawn on in provided color and thickness.
    '''
    # make a copy of the image
    draw_img = np.copy(image)
    # draw each bounding box on your image copy using cv2.rectangle()
    for ((x1, y1), (x2, y2)) in bboxes:
        draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255),
                                 thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes

# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
