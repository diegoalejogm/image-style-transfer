import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

IMAGES_PATH = './images'


def load_image(path, max_size=None, shape=None):
    '''
    Returns numpy array containing image data
    '''
    if max_size is not None and shape is not None:
        raise Exception('Either \'max_size\' or \'shape\' can be set.')

    img = Image.open('{}/{}'.format(IMAGES_PATH, path))

    # Resize image dimensions to same factor using max_size
    if max_size is not None:
        img.thumbnail(max_size, Image.ANTIALIAS)

    # Resize image dimensions to specified shape
    if shape is not None:
        img = img.resize(shape, Image.ANTIALIAS)

    # Convert image to numpy array
    arr = np.array(img)

    # Return numpy array
    return arr


def display_image(arr):
    plt.close()
    plt.figure()
    plt.imshow(arr)
