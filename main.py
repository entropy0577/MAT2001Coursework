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

# Function that solves part 1 the | symbol denotes a union type
def load_gs_img(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    print(img.size)
    img_as_arr = np.asarray(img.getdata()).reshape((img.size[1], img.size[0], 1))
    return img_as_arr

def create_columns(array, col):
    out = []
    for i in range(col):
        out.append(array[:,i])
    return out
    
# Problem 5 part 1 solved by using nested list comprehension as the numpy.in1d function
def create_emotion_matrix(img_arr: npt.ArrayLike, face_list: npt.ArrayLike) -> np.ndarray:
    return np.asarray([[1 if np.all(np.isin(i, j.reshape(123*4, 126, -1))) else 0 for i in img_arr] for j in face_list])
# Problem 4
def construct_gram(arr: npt.ArrayLike, func: Callable, sigma: float) -> np.ndarray:
    return np.asarray([[func(i, j, sigma) for i in arr] for j in arr])
# A function to calculate the kernal
def kernal(x: npt.ArrayLike, xprime: npt.ArrayLike, sigma: float) -> float:
    return math.exp( - 1 * (np.linalg.norm(x - xprime) ** 2 / 2 * (sigma ** 2)))
# Sln to problem 6
def classify(x: npt.ArrayLike, x_n: npt.ArrayLike, Z: npt.ArrayLike, sigma: float) -> np.ndarray:
    return Z @ np.asarray([kernal(i, x, sigma) for i in x_n])
# Returns the unit vector of the given vector
def normalize(n: npt.ArrayLike) -> np.ndarray:
    try: 
        return n / np.linalg.norm(n)
    except (ValueError, TypeError) as e:
        print(e)


# Checks if the code is being ran or imported
if __name__ == '__main__':
    # The images end up being slightly different resolutions, this corrects that
    test_as_arr = load_gs_img('test.png')[0:123, 0:126] / 255
    faces_as_arr = load_gs_img('faces.png') / 255
    # List comprehension solving part 2
    face_list_flat = np.asarray([[faces_as_arr[(26 + i*123):(149 + (i*123)), j*127:(126 + j * 127)].flatten()] for i in range(4) for j in range(6)])
    face_list = [[faces_as_arr[(26 + i*123):(149 + (i*123)), j*127:(126 + j * 127)] for i in range(4)] for j in range(6)]
    # Calculates the mean Problem 3
    mean = np.sum(face_list_flat) * 1/len(face_list_flat)
    # Calculates the variance by constructing a generator expression and then iterating through it, summing the values then dividing by the length of face_list
    variance = sum(np.linalg.norm(i - mean) ** 2 for i in face_list_flat) * 1/len(face_list_flat)
    gram = construct_gram(face_list_flat, kernal, variance)
    # Inverts gram
    graminverse = np.linalg.inv(gram)
    # Gets the emotion_matrix serving as Y for constructing Z
    face_mat = create_emotion_matrix(face_list_flat, np.asarray(face_list))

    print(face_mat)

    Z = np.asarray(face_mat) @ graminverse
    
    x = classify(test_as_arr.flatten(), face_list_flat, Z, variance)
    print(normalize(x))
