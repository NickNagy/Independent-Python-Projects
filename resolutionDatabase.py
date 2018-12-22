import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imresize
import random

data_path = 'D:\\flickr-image-dataset\\flickr30k_images\\'
img_folder = data_path + 'flickr30k_images\\'
training_folder = data_path + 'Training\\'
validation_folder = data_path + 'Validation\\'
testing_folder = data_path + 'Testing\\'

resolution = 2

y_dim = 512
x_dim = int(y_dim/resolution)

counter = 31783
for _, _, files in os.walk(img_folder):
    for file in files:
        which_folder = random.randint(0, 10)
        image = Image.open(img_folder + file)
        img_y = imresize(image, (y_dim, y_dim, 3))
        img_x = imresize(image, (x_dim, x_dim, 3))
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
