import cv2
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

def crop_piece(p, threshold_low=150, threshold_high=255):
    _, thresh = cv2.threshold(p, threshold_low, threshold_high, 0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_ixs = np.argsort(list(map(cv2.contourArea, contours)))
    # take the second largest contour
    contour_ix = contour_ixs[-2]
    contour = contours[contour_ix]
    x,y,w,h = cv2.boundingRect(contour)
    cropped_piece = p.copy()[y:y+h, x:x+w]
    return cropped_piece

def init_detector(name='sift'):
    if name == 'sift':
        return cv2.SIFT_create()
    if name == 'orb':
        return cv2.ORB_create()
    raise NotImplementedError

def init_matcher(name='flann'):
    if name == 'flann':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        return cv2.FlannBasedMatcher(index_params,search_params)
    if name == 'bf':
        return cv2.BFMatcher()
    raise NotImplemented

def init_draw_params(matches):
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # Ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return draw_params

def main(detector, matcher, img1, img2):
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)
    matches = matcher.knnMatch(des1,des2,k=2)
    draw_params = init_draw_params(matches)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return img3

if __name__ == "__main__":
    base_img = cv2.imread('./original/1.png', cv2.IMREAD_GRAYSCALE)

    params = dict(
        detector=init_detector('sift'), # sift, orb
        matcher=init_matcher('flann'),  # bf, flann
        img2=base_img,
        )
    callback = partial(main, **params)

    # TODO: Load pieces from webcam
    for _ in range(1):
        piece_ = cv2.imread('./sample.jpeg', cv2.IMREAD_GRAYSCALE)
        piece = crop_piece(piece_)
        img3 = callback(img1=piece)
        plt.imsave("match.jpg", img3)
        plt.imshow(img3,), plt.show()
