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


img_as_arr = load_gs_img('test.png')

print(img_as_arr.shape)

print(img_as_arr)

