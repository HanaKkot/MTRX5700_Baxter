import os
import sys
import numpy as np
import cv2

path_name = os.getcwd() + '/train'

image_size = 32
images = []
labels = []



def resize_image(image, height = image_size, width = image_size):
    return cv2.resize(image,(height, width))

def read_path(path_name):
    for files in os.listdir(path_name):
        full_path = os.path.join(path_name, files)
        image = resize_image(cv2.imread(full_path))
        images.append(image)
        labels.append(1)
    
    return images, labels

def load_data(path_name):
    images, labels = read_path(path_name)
    images = np.array(images)
    
    return images, labels
            
if __name__ == '__main__':
    read_path(path_name)
    print(len(images))
