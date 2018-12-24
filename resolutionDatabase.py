import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imresize
import random
import edgeDetector
import glob

data_path = 'D:\\flickr-image-dataset\\flickr30k_images\\'
img_folder = data_path + 'flickr30k_images\\'
training_folder = data_path + 'Training'
validation_folder = data_path + 'Validation\\'
testing_folder = data_path + 'Testing\\'

debug = False
sobel = True
laplace = True
canny = True
resolution = 2

y_dim = 512
x_dim = int(y_dim/resolution)

plt.gray()

def display(img):
    plt.imshow(img)
    plt.show()

def display_multiple(columns, images={}, titles={}):
    fig,ax = plt.subplots(1, columns, sharex=True, sharey=True)
    for i in range(0, columns):
        ax[i].imshow(images[i], aspect="auto")
        ax[i].set_title(titles[i])
    plt.show()

def generate_all_data(folder, counter = 31783):
    for _, _, files in os.walk(folder):
        for file in files:
            which_folder = random.randint(0, 10)
            image = Image.open(img_folder + file)
            img_y = imresize(image, (y_dim, y_dim, 3))
            img_x = imresize(image, (x_dim, x_dim, 3))
            if debug:
                fig,ax = plt.subplots(1, 2, sharex=False, sharey=False)
                ax[0].imshow(img_x)
                ax[1].imshow(img_y)
                plt.show()
            else:
                if which_folder < 6:
                    save_folder = training_folder
                elif which_folder < 8:
                    save_folder = validation_folder
                else:
                    save_folder = testing_folder
                np.save(save_folder + file + "_x.npy", img_x)
                np.save(save_folder + file + "_y.npy", img_y)
                np.save(save_folder + file + "_w.npy", np.ones(shape=(y_dim, y_dim, 1)))
            counter -= 1
            print(str(counter) + ": " + file)

def generate_weights(folder, counter):
    os.chdir(folder)
    for file in glob.glob('*_y.npy'):
        print(str(counter))
        img = np.asarray(Image.fromarray(np.load(file)).convert('L')) # awkward conversion type
        images = []
        titles = []
        images.append(img)
        titles.append("Original")
        if sobel:
            img_sobel = edgeDetector.sobel_detection(img)
            img_sobel_name = file[0:-6] + "_sobel"
            images.append(img_sobel)
            titles.append(img_sobel_name)
        if laplace:
            img_laplace = edgeDetector.laplace_detection(img)
            img_laplace_name = file[0:-6] + "_laplace"
            images.append(img_laplace)
            titles.append(img_laplace_name)
        if canny:
            img_canny = edgeDetector.canny_detection(img, 0)
            img_canny_name = file[0:-6] + "_canny"
            images.append(img_canny)
            titles.append(img_canny_name)
        if debug:
            display_multiple(columns=len(images), images=images, titles=titles)
        else:
            for i in range(0, len(images)):
                np.save(titles[i], images[i])
        counter-=1

generate_weights(training_folder, counter=17258)
generate_weights(validation_folder, counter=5874)
