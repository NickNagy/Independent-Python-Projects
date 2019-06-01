from PIL import Image
import os

DIRECTORY = "D://Dinosaurs//Images//"
SIZE = (256, 256)

def resize(img, size):
    return img.resize(size)

def resize_directory(dir, size):
    for _,_,files in os.walk(dir):
        for file in files:
            img = Image.open(dir + file).resize(size)
            img.save(dir + file)

if __name__ == "__main__":
    resize_directory(DIRECTORY, SIZE)