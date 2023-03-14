from PIL import Image
import numpy as np
import sympy as sym
import math
import matplotlib.pyplot as plt

img = Image.open("test.png").convert("L")

print(img.size)

img_as_arr = np.asarray(img.getdata()).reshape(img.size[0], img.size[1])

plt.imshow(img_as_arr[..., ::-1], cmap=plt.get_cmap('gray'), interpolation='Nearest')
plt.show()

print(img_as_arr)
