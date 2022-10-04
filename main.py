import cv2
import numpy as np
import sys

SATURATION_THRESHOLD = 35
"""
[For color pixel extraction]
Pixels with saturation greater than the specified saturation are processed as color pixels.
(min=0, max=255)
"""

COLOR_NOISE_THRESHOLD = 70
"""
[For black pixel extraction]
Pixels smaller than the specified color(RGB) are processed as noise that pixels not captured by color pixel extraction.
(min=0, max=255)
"""

CONTOUR_MAX_RGB_THRESHOLD = 180
"""
[For print pixel detection]
Contours containing pixels larger than the specified color(RGB) are treated as printed contents.
(min=0, max=255)
"""

CONTOUR_MEAN_RGB_THRESHOLD = 145
"""
[For print pixel detection]
Contours with an average color greater than the specified color(RGB) are treated as printed contents.
(min=0, max=255)
"""


def main(input_filename, output_filename):
    """
    Removes handwritten contents from the specified image.

    :param input_filename: input image name
    :param output_filename: output image name
    :return: None
    """

    # load image
    input_image = cv2.imread('./' + input_filename)
    if input_image is None:
        raise Exception('./' + input_filename + ': no such file or directory.')
    # remove uncolored contents
    color_image = remove_uncolored_pixels(input_image)
    # generate mask image for black pixels
    black_mask = generate_mask_for_black_pixels(input_image, color_image)
    # remove print content from mask
    handwritten_mask = remove_print_content_from_mask(black_mask)
    # inpaint input image by mask image
    output_img = inpaint(input_image, handwritten_mask)
    # save image
    cv2.imwrite('./' + output_filename, output_img)


def remove_uncolored_pixels(image):
    """
    Changes all pixels in the target image except those with color to white.

    Note: "Pixel with color" is defined as anything other than white or black.

    :param image: target image
    :return: image in which all pixels except colored pixels are changed to white
    """

    # BGR -> HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # turn all pixels except color pixels black (to ensure that black pixels are erased)
    low = np.array([0, 0, 0])
    high = np.array([180, SATURATION_THRESHOLD, 255])
    hsv_img_mask = cv2.inRange(hsv_img, low, high)
    hsv_img[hsv_img_mask > 0] = (0, 0, 0)

    # HSV -> BGR
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # whitens black pixels
    bgr_img[np.where((bgr_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return bgr_img


def generate_mask_for_black_pixels(base_img, color_img):
    """
    Generates a mask image showing only the black contents of the target image.

    :param base_img: target image
    :param color_img: target image with all but colored pixels removed
    :return: mask image showing only black pixels in the target image
    """

    # extract only black pixels in the image from the difference
    diff_img = cv2.absdiff(base_img, color_img)

    # remove any remaining noise (change all remaining color pixels to black)
    low = np.array([0, 0, 0])
    high = np.array([COLOR_NOISE_THRESHOLD, COLOR_NOISE_THRESHOLD, COLOR_NOISE_THRESHOLD])
    diff_img_mask = cv2.inRange(diff_img, low, high)
    diff_img[diff_img_mask > 0] = (0, 0, 0)

    return diff_img


def remove_print_content_from_mask(mask):
    """
    Removes only printed pixels from the mask image.

    :param mask: mask image containing printed and handwritten pixels
    :return: masked image with only handwritten pixels
    """

    # extract contours
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # fill only printed pixels with black to make a mask image of only handwritten pixels
    for contour in contours:
        # generates mask images for contours
        contour_mask = np.zeros(mask.shape[:2], np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # the border area of the outline is excluded from the mask,
        # because it tends to be darker in color than the center area and thus becomes noise.
        # mask = 255 - mask
        # cv2.drawContours(mask, [contour], -1, 255, 1)
        # mask = 255 - mask

        if is_print_content(mask, contour_mask):
            cv2.drawContours(mask, [contour], -1, 0, -1)

    return mask


def is_print_content(image, mask):
    """
    Determines if the specified masked area of the target image is printed.
    TODO: The accuracy may be improved by adding validation of standard deviation, mode, etc.

    :param image: target image
    :param mask: mask image
    :return: ture for printed pixels, false otherwise
    """

    # get the color closest to white within the contour
    gray_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_mask, mask=mask)

    # get the average color within a contour
    mean = cv2.mean(image, mask=mask)

    is_target_max = max_val > CONTOUR_MAX_RGB_THRESHOLD
    is_target_mean = mean[0] > CONTOUR_MEAN_RGB_THRESHOLD and mean[1] > CONTOUR_MEAN_RGB_THRESHOLD and mean[2] > CONTOUR_MEAN_RGB_THRESHOLD

    return is_target_max or is_target_mean


def inpaint(image, mask):
    """
    Inpaint the masked area of the target image.

    :param image: target image
    :param mask: mask image
    :return: inpainted image
    """
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return cv2.inpaint(image, gray_mask, 3, cv2.INPAINT_TELEA)


if __name__ == '__main__':
    try:
        args = sys.argv
        main(args[1], args[2])
    except Exception as e:
        print(e)
