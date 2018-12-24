import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.filters import threshold_mean
from math import atan

Gx_filter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
Gy_filter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
laplace_filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
laplace_filter_diag = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

filename = 'D:\\upResNet\\temp\\training\\test.jpg'

threshold = 225

plt.gray()
img = Image.open(filename).convert('L')
img_gauss = gaussian_filter(img, sigma=1.4)

imgx = convolve(img_gauss, Gx_filter)
imgy = convolve(img_gauss, Gy_filter)
imgL = convolve(img_gauss, laplace_filter)
imgLd = convolve(img_gauss, laplace_filter_diag)
imgxy = np.add(imgx, imgy)

def get_threshold(img):
    return img > threshold #threshold_mean(img)

def get_grad_angle(Gx, Gy):
    return atan(Gx,Gy)

def single_convolution():
    return 0

def canny_detection(gxy, thetaM, threshold):
    """
    :param gxy: input matrix, = |Gx| + |Gy|
    :param thetaM: input matrix, = arctan(Gy/Gx)
    :param threshold: threshold for edge definition
    """
    result = np.ndarray(shape=np.shape(img))
    #TODO: fix edge problem
    for i in range(0, gxy.shape[0]-3):
        for j in range(0, gxy.shape[1]-3):
            top_left = gxy[i][j]
            top_right = gxy[i+2][j]
            mid_top = gxy[i+1][j]
            mid_bot = gxy[i+1][j+2]
            mid_left = gxy[i][j+1]
            mid_right = gxy[i+2][j+1]
            bottom_left = gxy[i][j+2]
            bottom_right = gxy[i+2][j+2]
            center = gxy[i+1][j+1]
            theta = thetaM[i+1][j+1]
            if theta >= 22.5 and theta <= 67.5:
                if center > threshold and center > top_left and center > bottom_right:
                    result[i+1][j+1] = 1
                else:
                    result[i+1][j+1] = 0
            elif theta >= 67.5 and theta <= 112.5:
                if center > threshold and center > mid_top and center > mid_bot:
                    result[i+1][j+1] = 1
                else:
                    result[i+1][j+1] = 0
            elif theta >=112.5 and theta <= 157.5:
                if center > threshold and center > top_right and center > bottom_left:
                    result[i+1][j+1] = 1
                else:
                    result[i+1][j+1] = 0
            else:
                if center > threshold and center > mid_left and center > mid_right:
                    result[i+1][j+1] = 1
                else:
                    result[i+1][j+1] = 0
    # @ each point, check orientation of gradient and magnitude
    # determine if gradient is a maximum
    return result

imgx_thresh = get_threshold(imgx)
imgy_thresh = get_threshold(imgy)
imgL_thresh = get_threshold(imgL)
imgLd_thresh = get_threshold(imgLd)
imgxy_thresh = get_threshold(imgxy)

theta_matrix = np.arctan2(imgy,imgx)
img_canny = canny_detection(imgxy, theta_matrix, 0)
img_canny_thresh = canny_detection(imgxy, theta_matrix, threshold)

def display_all():
    fig,ax = plt.subplots(2,6,sharex=True,sharey=True)
    ax[0,0].imshow(imgx, aspect="auto")
    ax[0,0].set_title("Gx")
    ax[1,0].imshow(imgx_thresh, aspect="auto")
    ax[0,1].imshow(imgy, aspect="auto")
    ax[0,1].set_title("Gy")
    ax[1,1].imshow(imgy_thresh, aspect="auto")
    ax[0,2].imshow(imgxy, aspect="auto")
    ax[0,2].set_title("Sobel")
    ax[1,2].imshow(imgxy_thresh, aspect="auto")
    ax[0,3].imshow(imgL, aspect="auto")
    ax[0,3].set_title("Laplace")
    ax[1,3].imshow(imgL_thresh, aspect="auto")
    ax[0,4].imshow(imgLd, aspect="auto")
    ax[0,4].set_title("Laplace with diagonals")
    ax[1,4].imshow(imgLd_thresh, aspect="auto")
    ax[0,5].imshow(img_canny, aspect="auto")
    ax[0,5].set_title("Canny")
    ax[1,5].imshow(img_canny_thresh, aspect="auto")
    plt.show()

display_all()
