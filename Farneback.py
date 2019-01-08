"""
My implementation of the Farneback optical flow algorithm.
Algorithm works under the assumption of spatial consistency
The motion vector v b/w two images can be found from the differential relationship:
    multiply(transpose(gradientImage), v) = -(timeDerivativeImage)
If we have a matrix A of shape (, 2) where the columns represent x,y points in the gradient image, then mult(A.T, A)
creates a symmetrical matrix:
    [[Sigma(Ix*Ix), Sigma(Ix*Iy)],
    [Sigma(Ix*Iy), Sigma(Iy*Iy)]
If b is a row vector where each point represents a point in the time derivative image, then mult(A.T, b) is the dot
product b/w the gradient image points and the time derivative points:
    [[Sigma(x-distance*time], [Sigma(y-distance*time)]]

The results of my implementation are currently pretty noisy. I am trying to improve upon it by
    a) weighting pixel regions by optimizing e(x), defined in the original Farneback paper (Section 3.2)
    b) creating a pyramid of images of different resolutions
"""

from numpy import *
from scipy import signal
from scipy.ndimage import gaussian_filter
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

gx_filter = array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
gy_filter = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
t_filter = array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

debug = 0
factor = 1

plt.gray()

# TODO: implement grad. descent wrt weights in order to minimize e, then return w
def optimize_neighborhood(A, b, weights, iterations=3, alpha=1e-8):
    ATA = matmul(A.T, A)
    ATb = matmul(A.T, b)
    bTb = matmul(b.T, b)
    for i in range(0, iterations):
        e = weights*bTb - matmul(matmul(weights*linalg.pinv(ATA),weights*ATb).T, weights*ATb)
        #print("e(x): \n" + str(e))
        cost = bTb - 2*weights*matmul(matmul(linalg.pinv(ATA),ATb).T,ATb) # placeholder gradient, needs to be corrected
        if weights - alpha*cost > 0:
            weights -= alpha*cost
        if debug:
            print("Iteration: " + str(i) + "\n  e(x): " + str(e) + "\n cost: " + str(cost) + "\n weights: " + str(weights))
    return weights

def optical_flow(img1, img2, window_size, init_weights=1.0, iterations=1, poly_n=5, poly_sigma=1.1):
    #img1 = gaussian_filter(img1, sigma=1.4)
    #img2 = gaussian_filter(img2, sigma=1.4)
    Ix1 = signal.convolve2d(img1, gx_filter, boundary='symm', mode='same')
    Ix2 = signal.convolve2d(img2, gx_filter, boundary='symm', mode='same')
    Ix = 0.5*add(Ix1, Ix2)
    Iy1 = signal.convolve2d(img1, gy_filter, boundary='symm', mode='same')
    Iy2 = signal.convolve2d(img2, gy_filter, boundary='symm', mode='same')
    Iy = 0.5*add(Iy1, Iy2)
    It = signal.convolve2d(img2, t_filter, boundary='symm', mode='same') + \
         signal.convolve2d(img1, -t_filter, boundary='symm', mode='same')
    if debug:
        fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
        ax[0].imshow(img1, aspect="auto")
        ax[1].imshow(img2, aspect="auto")
        ax[2].imshow(Ix, aspect="auto")
        ax[3].imshow(Iy, aspect="auto")
        ax[4].imshow(It, aspect="auto")
        plt.show()
    w = int(window_size/2)
    u = zeros(img1.shape)
    v = zeros(img1.shape)
    i = w
    # TODO: fix indexing issues
    while i < img1.shape[0]-w:
        j = w
        while j < img1.shape[1] - w:
            Ax = Ix[i-w:i+w+1,j-w:j+w+1].flatten()
            Ay = Iy[i-w:i+w+1,j-w:j+w+1].flatten()
            b = -1 * It[i-w:i+w+1,j-w:j+w+1].flatten().T
            A = column_stack([Ax,Ay])
            try:
                weights = optimize_neighborhood(A, b, init_weights, iterations)
                d = factor*matmul(weights*linalg.pinv(matmul(A.T,A)), weights*matmul(A.T,b))
                u[i-w:i+w+1, j-w:j+w+1] = d[0]
                v[i-w:i+w+1, j-w:j+w+1] = d[1]
                if debug:
                    fig, ax = plt.subplots(2, 5, sharex=False, sharey=False)
                    px, py = (j - w), (i - w)
                    ax[0, 0].imshow(Ix, aspect="auto")
                    ax[0, 0].add_patch(Rectangle((px, py), w * 2, w * 2, color='r', fill=False))
                    ax[1, 0].imshow(Ix[i - w:i + w + 1, j - w:j + w + 1], aspect="auto")
                    ax[0, 1].imshow(Iy, aspect="auto")
                    ax[0, 1].add_patch(Rectangle((px, py), w * 2, w * 2, color='r', fill=False))
                    ax[1, 1].imshow(Iy[i - w:i + w + 1, j - w:j + w + 1], aspect="auto")
                    ax[0, 2].imshow(It, aspect="auto")
                    ax[0, 2].add_patch(Rectangle((px, py), w * 2, w * 2, color='r', fill=False))
                    ax[1, 2].imshow(It[i - w:i + w + 1, j - w:j + w + 1], aspect="auto")
                    ax[0, 3].imshow(u, aspect="auto")
                    ax[0, 3].add_patch(Rectangle((px, py), w * 2, w * 2, color='r', fill=False))
                    ax[1, 3].imshow(u[i-w: i + w + 1, j-w: j+w+1], aspect="auto")
                    ax[0, 4].imshow(v, aspect="auto")
                    ax[0, 4].add_patch(Rectangle((px, py), w * 2, w * 2, color='r', fill=False))
                    ax[1, 4].imshow(v[i-w: i + w + 1, j-w: j+w+1], aspect="auto")
                    plt.show()
            except linalg.LinAlgError:
                print("Regions must be square!")
            j += window_size
        i += window_size
    return stack([u, v], axis=2)

def test():
    img1 = random.rand(20,20)
    img2 = roll(img1, 1)#random.rand(20,20)
    print("Img 1: \n" + str(img1))
    print("Img 2: \n" + str(img2))
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(img1, aspect="auto")
    ax[1].imshow(img2, aspect="auto")
    plt.show()
    flow = optical_flow(img1,img2,window_size=3)
    draw_flow(img2, flow)

def draw_flow(gray_img, flow, step=16):
    h, w = gray_img.shape[:2]
    y, x = mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int) # creates a set of (y,x) points in equal
    # intervals (step) along image dimensions
    fx, fy = flow[y,x].T # creates two columns of vector data in a given interval that can be applied to show motion
    # expects flow to have same dimensions as gray_img
    lines = vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = int32(lines)
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0),-1)
    return vis

def show_vid_vectors():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = 0
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        my_flow = optical_flow(prev_gray, gray, window_size=15)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,None,0.5,3,15,3,5,1.2,0)
        prev_gray = gray
        my_flow_img = draw_flow(gray, my_flow)
        flow_img = draw_flow(gray, flow)
        cv2.imshow('My Algorithm', my_flow_img)
        cv2.imshow('CV Algorithm', flow_img)
        #cv2.imwrite(str(count) + ".jpg", flow_img)
        count += 1
        if cv2.waitKey(10) == 27:
            break

def record_and_save(num_imgs=20):
    cap = cv2.VideoCapture(0)
    for i in range(0, num_imgs):
        ret, img = cap.read()
        cv2.imwrite(str(i) + ".jpg", img)

def show_saved_vid_vectors(num_imgs = 20):
    for i in range(1, num_imgs):
        prev = cv2.cvtColor(cv2.imread(str(i-1) + ".jpg"), cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(cv2.imread(str(i) + ".jpg"), cv2.COLOR_BGR2GRAY)
        flow = optical_flow(prev, curr, window_size=15)
        flow_img = draw_flow(curr, flow)
        cv2.imwrite("Flow " + str(i) + ".jpg", flow_img)

#record_and_save()
#show_saved_vid_vectors()
#test()
show_vid_vectors()


