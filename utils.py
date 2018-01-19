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


def display_image(arr, format='HWC'):
    '''
    Displays image from numpy array using MatplotLib
    '''
    if format == 'CHW':
        arr = np.moveaxis(arr, 0, -1)
    plt.close()
    plt.figure()
    plt.imshow(arr)


def gram_matrices(list):
    '''
    Computes the gram matrices of list of 4-D (NCWH) Convolved Tensors.
    '''
    new_list = []
    for tensor in list:
        new_list.append(_gram_matrix(tensor))
    return new_list


def _gram_matrix(tensor):
    '''
    Computes the gram matrix of a 4-D (NCWH) Convolved Tensor.
    '''
    # Here N is num_tensors, no num_feature_maps as in the paper
    N = tensor.size(0)
    # Here C is num_feature_maps
    C = tensor.size(1)
    # Merge height and width dimensions into M: new shape: NCM
    tensor = tensor.view(N * C, -1)
    # Calculate transposed tensor
    tensor_T = tensor.transpose(0, 1)
    # Calculate Gram matrix as tensor_T * tensor
    tensor = torch.mm(tensor, tensor_T)
    return tensor


def style_loss(criterion, target_style, noise_style):
    '''
    Calculates the style loss given a loss criterion
    '''
    loss = 0
    num_layers = len(target_style)
    w = 1. / num_layers
    for i, _ in enumerate(target_style):
        style_i = target_style[i].detach()
        noise_i = noise_style[i]
        loss_i = criterion(noise_i, style_i) / w
        loss += loss_i
    return loss


def content_loss(criterion, target_content, noise_content):
    '''
    Calculates the content loss given a loss criterion
    '''
    loss = 0
    for i, _ in enumerate(target_content):
        content_i = target_content[i].detach()
        noise_i = noise_content[i]
        loss_i = criterion(noise_i, content_i)
        loss += loss_i
    return loss
