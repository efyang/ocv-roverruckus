import cv2
import numpy
import sys

render_resolution = (720, 480) # (1280, 720)

class OcvProcessor:
    def __init__(self, img):
        self.img = img
        self.blurred = self.box_blur(self.img, 10)
        self.grayblur = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2GRAY)
        self.rgb = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2RGB)
        self.yuv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2YUV)
        self.lab = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2LAB)
        self.hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)
        self.hsl = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HLS)
        self.ycrcb = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2YCrCb)
        # self.imshow("img", self.img)
        # self.imshow("rgb", self.rgb)
        # self.imshow("yuv", self.yuv)
        # self.imshow("lab", self.lab)
        # self.imshow("hsv", self.hsv)
        # self.imshow("hsl", self.hsl)
        # self.imshow("ycrcb", self.ycrcb)


    def process(self):
        sphere_contours = self.find_spheres()
        cube_contours = self.find_cubes()
        contoured = self.img.copy()
        cv2.drawContours(contoured, sphere_contours, -1, (255, 0, 0), 3)
        cv2.drawContours(contoured, cube_contours, -1, (0, 255, 0), 3)
        self.imshow("contoured", contoured)

    @staticmethod
    def watershed(img, sure_fg, sure_bg):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        markers = cv2.watershed(img, markers)
        return markers


    # TODO: use watershed to find overlapping contours. Then for each
    # watershed-found contour, remove any bounding contour from the original
    # list of contours. Then combine the list of contours.
    def find_spheres(self):
        rgbthresh = self.rgb_threshold(self.rgb, (190, 255), (178, 255), (170, 255))
        hsvthresh = self.hsv_threshold(self.hsv, (3, 48), (0, 72), (187, 255))
        basethresh = cv2.bitwise_or(rgbthresh, hsvthresh)
        masked = self.mask(self.grayblur, basethresh)
        normalized = self.normalize(masked, 0, 255)
        mul = cv2.multiply(masked, masked, 0.005)
        thresh1 = self.threshold(mul, 165, 255)
        erode1 = self.erode(thresh1, 5)
        closed = cv2.morphologyEx(erode1, cv2.MORPH_CLOSE, None, iterations = 20)
        distance = self.distance_transform(closed)
        thresh2 = self.threshold(distance, 40, 255)
        dilate1 = self.dilate(closed, 20)
        watershed_markers = self.watershed(closed, thresh2, dilate1)
        closed[watershed_markers == -1] = 0
        final = self.erode(closed, 1)
        contours1 = self.find_contours(final)
        contours1 = self.filter_contours(contours1, 1000, 0, 0, 100000, 0, 100000, (0, 100), 1000000, 0, 0, 5)
        return contours1

    def find_cubes(self):
        yuvthresh = self.yuv_threshold(self.yuv, (70, 255), (0, 100), (150, 255))
        labthresh = self.lab_threshold(self.lab, (100, 255), (100, 255), (160, 255))
        basethresh = cv2.bitwise_or(yuvthresh, labthresh)
        erode1 = self.erode(basethresh, 10)
        dilate1 = self.dilate(erode1, 10)
        closed = cv2.morphologyEx(dilate1, cv2.MORPH_CLOSE, None, iterations = 20)
        distance = self.distance_transform(closed)
        thresh1 = self.threshold(distance, 41, 255)
        dilate2 = self.dilate(closed, 10)
        watershed_markers = self.watershed(closed, thresh1, dilate2)
        dilate1[watershed_markers == -1] = 0
        final = self.erode(dilate1, 1)
        contours1 = self.find_contours(final)
        contours1 = self.filter_contours(contours1, 100, 0, 0, 100000, 0, 100000, (0, 100), 1000000, 20, 0, 5)
        return contours1

    @staticmethod
    def find_contours(input):
        mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        im2, contours, hierarchy = cv2.findContours(input, mode=mode, method=method)
        return contours

    @staticmethod
    def filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
        return output

    @staticmethod
    def distance_transform(input):
        h, w = input.shape[:2]
        dst = numpy.zeros((h, w), numpy.float32)
        cv2.distanceTransform(input, cv2.DIST_L2, 0, dst = dst)
        return numpy.uint8(dst)

    @staticmethod
    def erode(src, iterations):
        return cv2.erode(src, None, (-1, -1), iterations=iterations, borderType=cv2.BORDER_CONSTANT, borderValue=(-1))

    @staticmethod
    def dilate(src, iterations):
        return cv2.dilate(src, None, (-1, -1), iterations=iterations, borderType=cv2.BORDER_CONSTANT, borderValue=(-1))

    @staticmethod
    def threshold(src, thresh, max_val):
        return cv2.threshold(src, thresh, max_val, cv2.THRESH_BINARY)[1]

    @staticmethod
    def normalize(input, a, b):
        return cv2.normalize(input, None, a, b, cv2.NORM_MINMAX)

    @staticmethod
    def mask(input, mask):
        return cv2.bitwise_and(input, input, mask=mask)

    @staticmethod
    def box_blur(input, radius):
        ksize = 2 * radius + 1
        return cv2.blur(input, (ksize, ksize))

    @staticmethod
    def rgb_threshold(rgb, r, g, b):
        return cv2.inRange(rgb, (r[0], g[0], b[0]),  (r[1], g[1], b[1]))

    @staticmethod
    def lab_threshold(lab, l, a, b):
        return cv2.inRange(lab, (l[0], a[0], b[0]),  (l[1], a[1], b[1]))

    @staticmethod
    def hsl_threshold(hls, hue, sat, lum):
        return cv2.inRange(hls, (hue[0], lum[0], sat[0]),  (hue[1], lum[1], sat[1]))

    @staticmethod
    def hsv_threshold(hsv, hue, sat, val):
        return cv2.inRange(hsv, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def yuv_threshold(yuv, y, u, v):
        return cv2.inRange(yuv, (y[0], u[0], v[0]),  (y[1], u[1], v[1]))

    @staticmethod
    def imshow(name, img):
        cv2.imshow(name, cv2.resize(img, render_resolution))




infile = sys.argv[1]

cap = cv2.VideoCapture(infile)
while (cap.isOpened()):
    ret, frame = cap.read()
    # cv2.imshow("frame", frame)
    # print(frame)
    ocv = OcvProcessor(frame)
    ocv.process()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# img = cv2.imread(infile)
# img = cv2.resize(img, (1280, 720))

# ocv = OcvProcessor(img)
# ocv.process()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
