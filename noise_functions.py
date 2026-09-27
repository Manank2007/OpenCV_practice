import cv2 as cv
import numpy as np

#img=cv.imread("D:\CV_practice\practice_images\cats_3.jpg")

def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)  # keep valid pixel range
    return noisy
def add_salt_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white)
    ys = np.random.randint(0, h, num_pixels // 2)
    xs = np.random.randint(0, w, num_pixels // 2)
    noisy[ys, xs] = 255

    # Pepper (black)
    ys = np.random.randint(0, h, num_pixels // 2)
    xs = np.random.randint(0, w, num_pixels // 2)
    noisy[ys, xs] = 0

    return noisy

# noisy_image=add_gaussian_noise(img)
# #Gaussian Blurring:
# gaussian=cv.GaussianBlur(noisy_image,(5,5),1)
# #Contour detection
# canny_n_175_200=cv.Canny(noisy_image,220,300)
# #cv.imshow('canny_g',canny_g)
# canny_n205=cv.Canny(noisy_image,175,205)
# cv.imshow('canny_n(200)',canny_n_175_200)
# cv.imshow('canny_n(205)',canny_n205)
# cv.imshow('noisy',noisy_image)
# #cv.imshow('image',img)
# cv.waitKey(0)