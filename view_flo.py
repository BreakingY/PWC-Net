# Copyright (C) 2026 sunkx
# Licensed under the GNU General Public License v3.0
import numpy as np
import cv2

flow = cv2.readOpticalFlow('out.flo')

h, w = flow.shape[:2]
fx, fy = flow[..., 0], flow[..., 1]

mag, ang = cv2.cartToPolar(fx, fy)

hsv = np.zeros((h, w, 3), dtype=np.uint8)

ang = np.mod(ang, 2 * np.pi)

hue = np.zeros_like(ang, dtype=np.float32)

mask1 = ang < (np.pi / 2)
hue[mask1] = (ang[mask1] / (np.pi / 2)) * 30

mask2 = (ang >= np.pi / 2) & (ang < np.pi)
hue[mask2] = 30 + ((ang[mask2] - np.pi / 2) / (np.pi / 2)) * 90

mask3 = ang >= np.pi
hue[mask3] = 120 + ((ang[mask3] - np.pi) / np.pi) * 60

hsv[..., 0] = np.clip(hue, 0, 179).astype(np.uint8)

mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
hsv[..., 1] = mag_norm.astype(np.uint8)

hsv[..., 2] = 255

rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("flow semantic rgb", rgb)
cv2.waitKey(0)