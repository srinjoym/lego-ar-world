import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv2.imread('brick.jpg',0)
rotated_img = cv2.imread('brick_rotate.jpg', 0)

surf_img = cv2.xfeatures2d.SURF_create()
surf_rotated_img = cv2.xfeatures2d.SURF_create()

kp_img, des_img = surf_img.detectAndCompute(img,None)
kp_rot, des_rot = surf_rotated_img.detectAndCompute(rotated_img, None)

print(len(kp_img))
print(len(kp_rot))

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_img, des_rot, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_rot[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("MAP: Calculated scale difference: %.2f, "
                  "Calculated rotation difference: %.2f" %
                  (scaleRecovered, thetaRecovered))

    #deskew image
    im_out = cv2.warpPerspective(rotated_img, np.linalg.inv(M),
        (img.shape[1], img.shape[0]))

    # img2 = cv2.drawKeypoints(img,kp_img,None,(255,0,0),4)
    plt.imshow(im_out),plt.show()

else:
    self.log.warn("MAP: Not  enough  matches are found   -   %d/%d"
                  % (len(good), MIN_MATCH_COUNT))

# plt.figure()

# img3 = cv2.drawKeypoints(rotated_img,kp_rot,None,(255,0,0),4)
# plt.imshow(img3),plt.show()