from PIL import Image
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt
import math

# Authored by Rhys Ilott 
# Needs Pillow and numpy to run
# Type hints added for clarity

# Function that solves part 1 
def load_gs_img(path: str | Path):
    img = Image.open(path).convert('L')
    img_as_arr = np.asarray(img.getdata()).reshape((img.size[1], img.size[0], 1))
    return img_as_arr

# Problem 5 part 1 solved by using nested list comprehension as the numpy.in1d function
def create_emotion_matrix(img_arr: npt.ArrayLike, face_list: npt.ArrayLike) -> np.ndarray:
    return np.asarray([[1 if np.all(np.in1d(i, j)) else 0 for i in img_arr] for j in face_list.reshape(6, -1)[:,]])
# Problem 4
def construct_gram(arr: npt.ArrayLike, func: Callable) -> np.ndarray:
    return np.asarray([[func(i, j, 1) for i in arr] for j in arr])
# A function to calculate the kernal
def kernal(x: npt.ArrayLike, xprime: npt.ArrayLike, sigma: float) -> float:
    return math.exp( - 1 * (np.linalg.norm(x - xprime) ** 2 / 2 * (sigma ** 2)))

def classify(x: npt.ArrayLike, x_n: npt.ArrayLike, Z: npt.ArrayLike) -> np.ndarray:
    return Z @ np.asarray([kernal(i, x, 1) for i in x_n])



if __name__ == '__main__':
    # The images end up being slightly different resolutions, this correct that
    test_as_arr = load_gs_img('test.png')[0:123, 0:126]
    faces_as_arr = load_gs_img('faces.png')
    # List comprehension solving part 2
    face_list = [faces_as_arr[(26 + i*123):(149 + (i*123)), j*127:(126 + j * 127)].flatten() for i in range(4) for j in range(6)]
    # Calculates the mean Problem 3
    mean = np.sum(face_list) * 1/len(face_list)
    # Calculates the variance by constructing a generator expression and then iterating through it, summing the values then dividing by the length of face_list
    variance = sum(np.linalg.norm(i - mean) ** 2 for i in face_list) * 1/len(face_list)
    gram = construct_gram(face_list, kernal)

    graminverse = np.linalg.inv(gram)

    face_mat = create_emotion_matrix(face_list, np.asarray(face_list))

    Z = np.asarray(face_mat) @ graminverse

    x = classify(test_as_arr.flatten(), face_list, Z)
    print(x)
