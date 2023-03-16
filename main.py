from PIL import Image
import numpy as np
import numpy.typing as npt
from pathlib import Path
import matplotlib.pylab as plt
from typing import Callable
import math

# Authored by Rhys Ilott 
# Needs Pillow and numpy to run
# Type hints added for clarity

# Function that solves part 1 the | symbol denotes a union type
def load_gs_img(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    img_as_arr = np.asarray(img.getdata()).reshape((img.size[1], img.size[0], 1))
    return img_as_arr
    
# Problem 5 part 1 solved by using nested list comprehension and the numpy.isin function
def create_y_matrix(img_arr: npt.ArrayLike, face_list: npt.ArrayLike) -> np.ndarray:
    return np.asarray([[1 if np.all(np.isin(i, np.asarray(j).reshape(123*4, 126, -1))) else 0 for i in img_arr] for j in face_list])

# Problem 4
def construct_gram(arr: npt.ArrayLike, func: Callable, sigma: float) -> np.ndarray:
    return np.asarray([[func(i, j, sigma) for i in arr] for j in arr])

# A function to calculate the kernal
def kernal(x: npt.ArrayLike, xprime: npt.ArrayLike, sigma: float) -> float:
    return math.exp(-1 * (np.linalg.norm(x - xprime) ** 2 / (2 * sigma)))

# Sln to problem 6
def classify(x: npt.ArrayLike, x_n: npt.ArrayLike, Z: npt.ArrayLike, sigma: float) -> np.ndarray:
    return Z @ np.asarray([kernal(i, x, sigma) for i in x_n])

# Returns the unit vector of the given vector catching any divison exceptions
def normalize(n: npt.ArrayLike) -> np.ndarray:
    try: 
        return n / np.linalg.norm(n)
    except (ValueError, TypeError, RuntimeError) as e:
        print(e)

# Plots the classification vector with labels
def plot_output(x: npt.ArrayLike):
    fig, ax = plt.subplots()
    truncx = [round(i, 3) for i in x]
    emotions = ['Angriness', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Suprise']
    rects = ax.bar(emotions, x)
    for rect, label in zip(rects, truncx):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height,  label, ha='center', va='bottom')
    plt.show()    

# Checks if the code is being ran or imported
if __name__ == '__main__':
    # The images end up being slightly different resolutions, this corrects that
    test_as_arr = load_gs_img('test.png')[0:123, 0:126] / 255
    faces_as_arr = load_gs_img('faces.png') / 255
    # List comprehension solving part 2
    face_list_flat = np.asarray([faces_as_arr[(26 + i*123):(149 + (i*123)), j*127:(126 + j * 127)].flatten() for i in range(4) for j in range(6)])
    face_list = [[faces_as_arr[(26 + i*123):(149 + (i*123)), j*127:(126 + j * 127)] for i in range(4)] for j in range(6)]
    # Calculates the mean. Sln to Problem 3
    mean2 = np.mean(face_list_flat, axis=0)
    # Calculates the variance by constructing a generator expression and then iterating through it, summing the values then dividing by the length of face_list
    variance = sum((np.linalg.norm((mean2 - i)) ** 2) for i in face_list_flat) * 1/len(face_list_flat)
    # Calculates the gram matrix passing the kernal function as a paramater
    gram = construct_gram(face_list_flat, kernal, variance)
    # Inverts gram
    graminverse = np.linalg.inv(gram)
    # Gets the emotion_matrix serving as Y for constructing Z
    face_mat = create_y_matrix(face_list_flat, np.asarray(face_list))
    # Calculates Z identical to using np.matmult
    Z = face_mat @ graminverse
    # Classifies the test image against the training data using the gram matrix, and then normalizes it 
    x = normalize(classify(test_as_arr.flatten(), face_list_flat, Z, variance))
    # Prints the 6 dimensional vector
    print(x)
    # Plots the 6 dimensional vector
    plot_output(x)
    # If test.png was one of the images showing digust the output would be [0, 1, 0, 0, 0, 0]