#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import imutils


class MotionTracker:
    def __init__(self):
        self.bridge = CvBridge()
        self.min_area = 50
        self.max_area = 50
        self.last_frame = None
        self.kernel = np.ones((5, 5), np.uint8)
        self.queue = []
        self.queue_size = 0

        rospy.Subscriber('/usb_cam/image_raw', Image, self._callback)

    def convex_draw(self, contours, image):
        hull = []
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 40:
                continue
            (x, y, w, h) = cv2.boundingRect(contours[i])
            solidity = cv2.contourArea(contours[i]) / (w * h)

            if solidity < 0.1:
                continue
            hull.append(cv2.convexHull(contours[i], False))
        for i in range(len(hull)):
            cv2.drawContours(image,
                             hull,
                             i,
                             (0, 0, 255), 2, 8)

    def _callback(self, Image):
        image = self.bridge.imgmsg_to_cv2(Image, "bgr8")
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,
                               120,
                               255,
                               cv2.THRESH_BINARY_INV)[1]

        thresh = cv2.morphologyEx(thresh,
                                  cv2.MORPH_OPEN,
                                  self.kernel)
        if self.last_frame is None:
            self.last_frame = thresh
            return
        frame_delta = cv2.absdiff(self.last_frame, thresh)
        self.last_frame = thresh

        if len(self.queue) > 20:
            self.queue.pop(0)
        self.queue.append(frame_delta)
        for index in range(len(self.queue)):
            thresh = cv2.bitwise_or(thresh, self.queue[index])
        frame_delta = cv2.absdiff(self.last_frame, thresh)
        contours = cv2.findContours(frame_delta,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        self.convex_draw(contours, image)

        cv2.imshow("frame_delta", frame_delta)
        cv2.imshow("Motion-Tracker", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node('motion_tracker_node')
    motion = MotionTracker()
    rospy.spin()
