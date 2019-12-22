import cv2
import imutils


class AspectAwarePreProcessor:
    def __init__(self, height, width, inter=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.inter = inter

    def preprocess(self, image):
        h, w = image.shape[:2]
        dh, dw = 0, 0
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dh = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dw = int((image.shape[1] - self.width) / 2.0)
        # now that our images have been resized, we need to re-grab the width and height,
        # followed by performing  the crop
        h, w = image.shape[:2]
        image = image[dh:h - dh, dw:w - dw]
        # Finally, resize the image to the provided  spatial dimensions to ensure the output image is always a fixed
        # dimensions
        return cv2.resize(image, (self.height, self.width), interpolation=self.inter)
