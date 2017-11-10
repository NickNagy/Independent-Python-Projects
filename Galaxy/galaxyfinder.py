from PIL import Image
from pylab import *
import kmeans
import matplotlib

original_image = Image.open('example.png')
im = array(original_image.convert('L'))
m, n = im.shape  # m = rows, n = columns
white = []

for i in range(m):
    for j in range(n):
        if im[i][j] == 255:
            white.append([i, j])

km = kmeans.KMeans(k_max=10)
center_points = km.k_fits(white)  # find k number of cluster centers of white space, where k is optimal

# each cluster gets a color --> change all the white in the image to those colors
cluster_im = im