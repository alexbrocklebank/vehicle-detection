import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')
#image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
            'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']

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


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    imcopy = np.copy(img)
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    for template in template_list:
        # Read in templates one by one
        temp_img = mpimg.imread(template)
        temp_h, temp_w, _ = temp_img.shape
        # Use cv2.matchTemplate() to search the image
        #     using whichever of the OpenCV search methods you prefer
        result = cv2.matchTemplate(imcopy, temp_img, cv2.TM_SQDIFF_NORMED)
        # Use cv2.minMaxLoc() to extract the location of the best match
        _, _, min, _ = cv2.minMaxLoc(result)
        max = (min[0] + temp_w, min[1] + temp_h)
        # Determine bounding box corners for the match
        bbox_list.append((min, max))
    # Return the list of bounding boxes
    return bbox_list

bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
