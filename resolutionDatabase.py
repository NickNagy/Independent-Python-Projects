import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imresize

data_path = 'D:\\flickr-image-dataset\\flickr30k_images\\flickr30k_images\\'

resolution = 2

total_width = 0
total_height = 0
total_images = 0

images = OrderedDict()

for _, _, files in os.walk(data_path):
    for file in files:
        total_images += 1
        print(file)
        image = Image.open(data_path + file)
        total_width += np.shape(image)[0]
        total_height += np.shape(image)[1]
        images[file] = image

print(total_width / total_images)
print(total_height / total_images)

avg_width = int(total_width/total_images)
avg_height = int(total_height/total_images)

for image in images:
    img_y = imresize(image, (avg_width, avg_height, 3))
    img_x = imresize(img_y, (np.shape(img_y)[0]/resolution, np.shape(img_y)[1]/resolution, 3))
    Image.save