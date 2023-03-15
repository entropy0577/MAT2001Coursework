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


# Using list comprehension
def variance_of_arr(array):
    return np.asarray([np.linalg.norm(i - np.mean(i)) / len(array) for i in array])

test_as_arr = load_gs_img('test.png')
faces_as_arr = load_gs_img('faces.png')
# A generator expression solving part 2
face_list = [faces_as_arr[(26 + i*123):(149 + (i*123)), j*127:(126 + j * 127)].flatten() for i in range(4) for j in range(6)]
# A generator expression containing the variances 
variance_gen = (np.linalg.norm(i - np.mean(i) / 24) for i in face_list)

gram = [np.dot(i, j) for i in face_list for j in face_list]

print(gram)