import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch

IMAGES_PATH = './images'


def load_image(path, max_size=None, shape=None, display=True):
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

    # Display image
    if display:
        display_image(arr)

    # Return numpy array
    return arr


def image_to_var(arr):
    '''
    Convert 3D numpy image to 4D PyTorch Variable
    '''
    # Reformat array WHC to CWH
    arr = np.moveaxis(arr, 2, 0)
    # Convert to float Tensor
    tensor = torch.from_numpy(arr).float()
    # Reformat add 4th dimension N: CWH to NCWH
    tensor = tensor.unsqueeze(0)
    # Convert to Variable
    return Variable(tensor)


def display_image(arr):
    plt.close()
    plt.figure()
    plt.imshow(arr)
