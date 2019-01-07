from numpy import *
from scipy import signal
import cv2
from matplotlib import pyplot as plt

gx_filter = array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
gy_filter = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
t_filter = array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])

debug = 0
factor = 1 # -0.5

#(A.T*A)d = A.T*b
def optical_flow(img1, img2, window_size, iterations=None, poly_n=5, poly_sigma=1.1):
    Ix = signal.convolve2d(img1, gx_filter, boundary='symm', mode='same')
    Iy = signal.convolve2d(img1, gy_filter, boundary='symm', mode='same')
    It = signal.convolve2d(img2, t_filter, boundary='symm', mode='same') + \
         signal.convolve2d(img1, -t_filter, boundary='symm', mode='same')
    if debug:
        print("Ix: \n" + str(Ix))
        print("Iy: \n" + str(Iy))
        print("It: \n" + str(It))
    w = int(window_size/2)
    u = zeros(img1.shape)#.flatten()#(img1.shape[0])
    v = zeros(img1.shape)#.flatten()#(img1.shape[1])
    for i in range(w, img1.shape[0]-w):
        for j in range(w, img1.shape[1]-w):
            Ax = Ix[i-w:i+w+1,j-w:j+w+1].flatten()
            Ay = Iy[i-w:i+w+1,j-w:j+w+1].flatten()
            b = It[i-w:i+w+1,j-w:j+w+1].flatten()
            A = vstack([Ax,Ay]).T
            #print(matmul(A.T, A))
            if debug:
                print("****** ITERATION " + str(i-w) + "******")
                print("Ax: \n" + str(Ax))
                print("Ay: \n" + str(Ay))
                print("b: \n" + str(b))
            try:
                d = factor*matmul(linalg.pinv(matmul(A.T,A)), matmul(A.T,b))
                u[i-w:i+w+1, j-w:j+w+1] = d[0]
                v[i-w:i+w+1, j-w:j+w+1] = d[1]
                #u[i*j] = factor*#matmul(linalg.inv(matmul(Ax.T, Ax)), matmul(Ax.T, b))#d[0]
                #v[i*j] = factor*matmul(linalg.inv(matmul(Ay.T, Ay)), matmul(Ay.T, b))#d[1]
                if debug:
                    print("U: \n" + str(u))
                    print("V: \n" + str(v))
            except linalg.LinAlgError:
                print("Regions must be square!")
    return stack([u, v], axis=2) #vstack([u,v])

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
    #print("********************")
    #print("U: \n" + str(u))
    #print("V: \n" + str(v))
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
    #plt.imshow(vis)
    #plt.show()
    return vis

def show_vid_vectors():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # TODO: expects flow to have same shape as prev_gray
        flow = optical_flow(prev_gray, gray, window_size=400)
        prev_gray = gray
        cv2.imshow('Optical Flow', draw_flow(gray, flow))
        if cv2.waitKey(10) == 27:
            break

show_vid_vectors()
#test()
