import cv2
import numpy
import math
from enum import Enum

class GripPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """

    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__blur_type = BlurType.Box_Blur
        self.__blur_radius = 47.74774774774775

        self.blur_output = None

        self.__hsv_threshold_input = self.blur_output
        self.__hsv_threshold_hue = [89.83050847457629, 128.6631016042781]
        self.__hsv_threshold_saturation = [0.0, 36.64753335401801]
        self.__hsv_threshold_value = [161.251422590741, 255.0]

        self.hsv_threshold_output = None

        self.__rgb_threshold_input = self.blur_output
        self.__rgb_threshold_red = [115.09013128480268, 255.0]
        self.__rgb_threshold_green = [143.47183270332883, 255.0]
        self.__rgb_threshold_blue = [167.1186440677966, 255.0]

        self.rgb_threshold_output = None

        self.__cv_bitwise_or_src1 = self.hsv_threshold_output
        self.__cv_bitwise_or_src2 = self.rgb_threshold_output

        self.cv_bitwise_or_output = None

        self.__cv_erode_src = self.cv_bitwise_or_output
        self.__cv_erode_kernel = None
        self.__cv_erode_anchor = (-1, -1)
        self.__cv_erode_iterations = 10.0
        self.__cv_erode_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_bordervalue = (-1)

        self.cv_erode_output = None

        self.__cv_dilate_0_src = self.cv_erode_output
        self.__cv_dilate_0_kernel = None
        self.__cv_dilate_0_anchor = (-1, -1)
        self.__cv_dilate_0_iterations = 30.0
        self.__cv_dilate_0_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_0_bordervalue = (-1)

        self.cv_dilate_0_output = None

        self.__distance_transform_input = self.cv_dilate_0_output
        self.__distance_transform_type = cv2.DIST_L2
        self.__distance_transform_mask_size = 0

        self.distance_transform_output = None

        self.__cv_threshold_src = self.distance_transform_output
        self.__cv_threshold_thresh = 30.0
        self.__cv_threshold_maxval = 255.0
        self.__cv_threshold_type = cv2.THRESH_BINARY

        self.cv_threshold_output = None

        self.__cv_dilate_1_src = self.cv_dilate_0_output
        self.__cv_dilate_1_kernel = None
        self.__cv_dilate_1_anchor = (-1, -1)
        self.__cv_dilate_1_iterations = 10.0
        self.__cv_dilate_1_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_1_bordervalue = (-1)

        self.cv_dilate_1_output = None


    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step Blur0:
        self.__blur_input = source0
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.blur_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step RGB_Threshold0:
        self.__rgb_threshold_input = self.blur_output
        (self.rgb_threshold_output) = self.__rgb_threshold(self.__rgb_threshold_input, self.__rgb_threshold_red, self.__rgb_threshold_green, self.__rgb_threshold_blue)

        # Step CV_bitwise_or0:
        self.__cv_bitwise_or_src1 = self.hsv_threshold_output
        self.__cv_bitwise_or_src2 = self.rgb_threshold_output
        (self.cv_bitwise_or_output) = self.__cv_bitwise_or(self.__cv_bitwise_or_src1, self.__cv_bitwise_or_src2)

        # Step CV_erode0:
        self.__cv_erode_src = self.cv_bitwise_or_output
        (self.cv_erode_output) = self.__cv_erode(self.__cv_erode_src, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)

        # Step CV_dilate0:
        self.__cv_dilate_0_src = self.cv_erode_output
        (self.cv_dilate_0_output) = self.__cv_dilate(self.__cv_dilate_0_src, self.__cv_dilate_0_kernel, self.__cv_dilate_0_anchor, self.__cv_dilate_0_iterations, self.__cv_dilate_0_bordertype, self.__cv_dilate_0_bordervalue)

        # Step Distance_Transform0:
        self.__distance_transform_input = self.cv_dilate_0_output
        (self.distance_transform_output) = self.__distance_transform(self.__distance_transform_input, self.__distance_transform_type, self.__distance_transform_mask_size)

        # Step CV_Threshold0:
        self.__cv_threshold_src = self.distance_transform_output
        (self.cv_threshold_output) = self.__cv_threshold(self.__cv_threshold_src, self.__cv_threshold_thresh, self.__cv_threshold_maxval, self.__cv_threshold_type)

        # Step CV_dilate1:
        self.__cv_dilate_1_src = self.cv_dilate_0_output
        (self.cv_dilate_1_output) = self.__cv_dilate(self.__cv_dilate_1_src, self.__cv_dilate_1_kernel, self.__cv_dilate_1_anchor, self.__cv_dilate_1_iterations, self.__cv_dilate_1_bordertype, self.__cv_dilate_1_bordervalue)


    @staticmethod
    def __blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        if(type is BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif(type is BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif(type is BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __rgb_threshold(input, red, green, blue):
        """Segment an image based on color ranges.
        Args:
            input: A BGR numpy.ndarray.
            red: A list of two numbers the are the min and max red.
            green: A list of two numbers the are the min and max green.
            blue: A list of two numbers the are the min and max blue.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        return cv2.inRange(out, (red[0], green[0], blue[0]),  (red[1], green[1], blue[1]))

    @staticmethod
    def __cv_bitwise_or(src1, src2):
        """Computes the per channel or of two images.
        Args:
            src1: A numpy.ndarray.
            src2: A numpy.ndarray.
        Returns:
            A numpy.ndarray the or of the two mats.
        """
        return cv2.bitwise_or(src1, src2)

    @staticmethod
    def __cv_erode(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of lower value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for erosion. A numpy.ndarray.
           iterations: the number of times to erode.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after erosion.
        """
        return cv2.erode(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __distance_transform(input, type, mask_size):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.array.
            type: Opencv enum.
            mask_size: The size of the mask. Either 0, 3, or 5.
        Returns:
            A black and white numpy.ndarray.
        """
        h, w = input.shape[:2]
        dst = numpy.zeros((h, w), numpy.float32)
        cv2.distanceTransform(input, type, mask_size, dst = dst)
        return numpy.uint8(dst)

    @staticmethod
    def __cv_threshold(src, thresh, max_val, type):
        """Apply a fixed-level threshold to each array element in an image
        Args:
            src: A numpy.ndarray.
            thresh: Threshold value.
            max_val: Maximum value for THRES_BINARY and THRES_BINARY_INV.
            type: Opencv enum.
        Returns:
            A black and white numpy.ndarray.
        """
        return cv2.threshold(src, thresh, max_val, type)[1]

    @staticmethod
    def __cv_dilate(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of higher value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for dilation. A numpy.ndarray.
           iterations: the number of times to dilate.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after dilation.
        """
        return cv2.dilate(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)


BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')
img = cv2.imread("/home/efyang/GRIP/FTC  RR2 Photo Library/Mineral_Photos/Samsung s5 Camera App/20180910_095634.jpg", 1)
img = cv2.resize(img, (1280, 720))
grip = GripPipeline()
grip.process(img)
im2 = cv2.cvtColor(grip.cv_dilate_0_output, cv2.COLOR_GRAY2RGB)
cv2.imshow("img", grip.cv_threshold_output)
sure_fg = numpy.uint8(grip.cv_threshold_output)
sure_bg = numpy.uint8(grip.cv_dilate_1_output)
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow("uk", cv2.resize(unknown, (1280, 720)))
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0
markers = cv2.watershed(im2, markers)
img[markers == -1] = [255,0,0]
cv2.imshow("test", cv2.resize(img, (1280, 720)))
cv2.waitKey(0)
cv2.destroyAllWindows()
