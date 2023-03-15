from PIL import Image
import numpy as np
import sympy as sym
import math
import matplotlib.pyplot as plt


# Function that solves part 1
def load_gs_img(path):
    img = Image.open(path).convert('L')
    img_as_arr = np.asarray(img.getdata()).reshape((img.size[1], img.size[0], 1))
    return img_as_arr


# Solves part 2. Uses list comprehension to create a new list
def get_faces(hindex, vindex, img_as_arr):
    yield (img_as_arr[(26 + i*123):(149 + (i*123)), j*127:(126 + j * 127)].flatten() for i in range(hindex)
            for j in range(vindex))

# Using list comprehension
def variance_of_arr(array):
    return np.asarray([np.linalg.norm(i - np.mean(i)) / len(array) for i in array])



test_as_arr = load_gs_img('test.png')
faces_as_arr = load_gs_img('faces.png')
face_list = get_faces(4, 6, faces_as_arr)

print(face_list)
