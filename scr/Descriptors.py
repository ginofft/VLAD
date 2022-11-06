import cv2

def compute_SIFT(img):
  sift = cv2.SIFT_create()
  kp, des = sift.detectAndCompute(img,None)
  return des