# rgbd_pose/detection.py

import cv2
import numpy as np


class DetectionResult:

    def __init__(self, center_px, bbox, contour_area, mask):
        self.center_px = center_px
        self.bbox = bbox
        self.contour_area = contour_area
        self.mask = mask


class ObjectDetector:

    def __init__(
        self,
        hsv_lower=(100, 80, 80),
        hsv_upper=(130, 255, 255),
        min_area=200.0,
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area

    def detect(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB image")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        return DetectionResult(
            center_px=(cx, cy),
            bbox=(x, y, w, h),
            contour_area=area,
            mask=mask,
        )
