# Copyright (C) 2026 sunkx
# Licensed under the GNU General Public License v3.0
import numpy as np
import cv2

flow = cv2.readOpticalFlow('out.flo')

# 转 HSV 显示
h, w = flow.shape[:2]
fx, fy = flow[..., 0], flow[..., 1]

mag, ang = cv2.cartToPolar(fx, fy)

hsv = np.zeros((h, w, 3), dtype=np.uint8)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 1] = 255
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("flow rgb", rgb)
cv2.waitKey(0)