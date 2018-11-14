import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def color_mask(img):
    light_blue = (90, 100, 20) # masks for blue color
    dark_blue = (120, 255, 255)

    # light_yellow = (20, 100, 100)
    # dark_yellow = (30, 255, 255)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv_img, light_blue, dark_blue)

def extract_keypoints(img):
    mask = color_mask(img)
    masked_img = cv2.bitwise_and(img, img, mask=mask)[:,:,0]
    # surf_img = cv2.xfeatures2d.SURF_create()
    # kp_img, des_img = surf_img.detectAndCompute(masked_img,None)
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    kp_img, des_img = orb.compute(img, kp)
    return kp_img, des_img

def find_homography(kp1, kp2, des1, des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                            ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                            ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        ss = M[0, 1]
        sc = M[0, 0]
        scaleRecovered = math.sqrt(ss * ss + sc * sc)
        thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        print("MAP: Calculated scale difference: %.2f, "
                    "Calculated rotation difference: %.2f" %
                    (scaleRecovered, thetaRecovered))

        return M, mask, good

def scale_image(img, scale_percent = 30):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def crop_to_largest_blob(img):
    mask = color_mask(img)
    # gray = cv2.bitwise_and(img, img, mask=mask)[:,:,0]
    # # gray = mask_image(img)
    # _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    max_area = -1
    max = None
    for contour in contours:
        if cv2.contourArea(contour) > max_area:
            max_area = cv2.contourArea(contour)
            max = contour

    x,y,w,h = cv2.boundingRect(max)
    crop = img[y:y+h,x:x+w]
    return crop

def main():
    ground_truth_img = cv2.imread('brick.jpg')
    ground_truth_img = crop_to_largest_blob(ground_truth_img)

    print(len(ground_truth_img))
    kp_gt, des_gt = extract_keypoints(ground_truth_img)

    cap = cv2.VideoCapture('brick_rotating.mp4')

    while(cap.isOpened()):
        start = cv2.getTickCount()
        ret, frame = cap.read()
        frame = crop_to_largest_blob(frame)
        kp_frame, des_frame = extract_keypoints(frame)

        # plt.subplot(1,2,1), plt.imshow(masked_img)
        # plt.subplot(1,2,2), plt.imshow(masked_rotated_img)
        # plt.show()

        # print(len(kp_gt))
        # print(len(kp_frame))

        M, mask, good = find_homography(kp_gt, kp_frame, des_gt, des_frame)
        src_kp = []
        dest_kp = []
        for m in good:
            src_kp.append(kp_gt[m.queryIdx])
            dest_kp.append(kp_frame[m.trainIdx])

        masked_img_keypoints = cv2.drawKeypoints(ground_truth_img,src_kp,None,(255,0,0),4)
        cv2.imshow("1",masked_img_keypoints)

        masked_rot_img_keypoints = cv2.drawKeypoints(frame,dest_kp,None,(255,0,0),4)
        cv2.imshow("2",masked_rot_img_keypoints)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # plt.show()

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)
        print(fps)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()